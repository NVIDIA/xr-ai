# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic voice routing and streaming tests without downloaded models or recorded audio."""

from __future__ import annotations

import asyncio
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import uvicorn
import yaml
from pipecat.frames.frames import CancelFrame, InputAudioRawFrame, InterruptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from speaker_stt.__main__ import _build_app, _Server
from speaker_stt._selection import _Selection
from websockets.asyncio.server import serve
from xr_ai_models._speaker_stream import _SpeakerStream
from xr_ai_voice._frames import (
    GatedQueryFrame,
    ParticipantLeftFrame,
    _SpeakerEnrollmentFrame,
    _SpeakerTranscriptionFrame,
)
from xr_ai_voice._pipeline import _build_voice_pipeline
from xr_ai_voice._processors.io import _VoiceIOProcessor
from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor
from xr_ai_voice._processors.vad_stt import VadConfig, VadSttProcessor
from xr_ai_voice._processors.voice_gate import VoiceGateProcessor
from xr_ai_voice._speaker_client import _select_speaker_asr, _SpeakerClient
from xr_ai_voice._transport import HubVoiceTransport
from xr_ai_voicegate import load_voice_gate_config
from xr_ai_voicegate._speaker import _SpeakerConfig


@pytest.mark.asyncio
async def test_empty_speaker_utterance_invalidates_control_without_dropping_next_query(tmp_path):
    cfg = _voice_cfg(tmp_path)
    selection = _Selection(cfg._speaker)
    processor = _SpeakerSttProcessor(cfg=cfg._speaker)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate.push_frame = AsyncMock()

    async def to_gate(frame, *_args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    processor.push_frame = to_gate

    async def utterance(text, pts_us):
        selection._activity({7}, 0.2, pts_us)
        for event in selection._finish(text):
            await processor._event("wearer", event)

    try:
        await utterance(cfg._speaker.start_phrase, 0)
        gate.push_frame.reset_mock()
        await utterance("hey agent", 1_000_000)
        await utterance("", 2_000_000)
        await utterance("let's start talking", 3_000_000)
        queries = [c.args[0] for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]
        assert [q.text for q in queries] == ["let's start talking"]
        assert selection.owner == 7 and "wearer" in gate._conversation_active
    finally:
        await processor.cleanup()
        await gate.cleanup()


@pytest.mark.asyncio
async def test_overlapping_start_phrase_enrolls_its_speaker_and_routes_only_owner_queries(tmp_path):
    cfg = _voice_cfg(tmp_path)
    selection = _Selection(cfg._speaker)
    processor = _SpeakerSttProcessor(cfg=cfg._speaker)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._use_speaker_asr = True
    gate.push_frame = AsyncMock()

    async def to_gate(frame, *_args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    processor.push_frame = to_gate

    async def deliver(events):
        for event in events:
            await processor._event("wearer", event)

    try:
        # A's ongoing ordinary speech cannot reserve enrollment. The service
        # recognizes B's phrase in B's own conditioned stream during overlap.
        selection._activity({1}, 0.2, 0)
        selection._activity({1, 7}, 0.2, 200_000)
        await deliver(selection._enroll_completed([
            (1, "what time is it", 0, False),
            (7, cfg._speaker.start_phrase, 200_000, False),
        ]))
        assert selection.owner == 7
        assert "wearer" in processor._enrolled and "wearer" in gate._conversation_active

        # Later phrases cannot take ownership away from the enrolled speaker.
        assert not selection._enroll_completed([(1, cfg._speaker.start_phrase, 800_000, False)])
        assert selection.owner == 7
        gate.push_frame.reset_mock()
        selected, events, _ = selection._activity({1}, 0.2, 1_000_000)
        assert selected is None
        await deliver(events)
        assert not gate.push_frame.called

        selected, events, _ = selection._activity({1, 7}, 0.2, 1_200_000)
        assert selected == 7
        await deliver(events)
        await deliver(selection._finish("look around"))
        queries = [c.args[0] for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]
        assert len(queries) == 1
        assert queries[0].text == "look around" and queries[0].participant_id == "wearer"
        assert queries[0].pts_us == 1_200_000
    finally:
        await processor.cleanup()
        await gate.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("speakers", [(1, 7), (7, 1)])
async def test_same_step_start_phrase_tie_does_not_enable_voice_gate(tmp_path, speakers):
    cfg = _voice_cfg(tmp_path)
    selection = _Selection(cfg._speaker)
    processor = _SpeakerSttProcessor(cfg=cfg._speaker)
    processor.push_frame = AsyncMock()
    try:
        events = selection._enroll_completed([
            (speaker, cfg._speaker.start_phrase, 0, False) for speaker in speakers
        ])
        for event in events:
            await processor._event("wearer", event)
        assert selection.owner is None and not processor._enrolled
        await processor._event("wearer", {
            "kind": "transcript", "text": "look around", "pts_us": 1_000_000, "speaker_id": 1,
        })
        assert not processor.push_frame.called
    finally:
        await processor.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_op", ["open", "audio"])
async def test_cancelled_request_closes_connection_and_releases_session(cancel_op):
    allocated, release_reply = asyncio.Event(), asyncio.Event()
    released = asyncio.Event()
    sessions = set()

    async def handler(socket):
        await socket.recv()
        sessions.add(socket)
        try:
            if cancel_op == "open":
                allocated.set()
                await release_reply.wait()
            await socket.send(json.dumps({"service": "xr-ai-speaker-stt", "protocol": 3}))
            await socket.recv()
            if cancel_op == "audio":
                allocated.set()
                await release_reply.wait()
            await socket.send(json.dumps({"events": []}))
            await socket.wait_closed()
        finally:
            sessions.discard(socket)
            released.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = _SpeakerClient(_SpeakerConfig(base_url=f"http://127.0.0.1:{port}", timeout_s=1))
        feed = asyncio.create_task(client._feed("wearer", bytes(640), 1))
        try:
            await asyncio.wait_for(allocated.wait(), timeout=1)
            assert len(sessions) == 1
            feed.cancel()
            await asyncio.gather(feed, return_exceptions=True)
            release_reply.set()
            await client._close()
            await asyncio.wait_for(released.wait(), timeout=1)
            assert not sessions and not client._sessions
        finally:
            release_reply.set()
            feed.cancel()
            await asyncio.gather(feed, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_stream_close_still_disconnects_the_transport():
    closing = asyncio.Event()

    async def delayed_close():
        closing.set()
        await asyncio.Future()

    transport = SimpleNamespace(abort=Mock())
    stream = _SpeakerStream("http://127.0.0.1:8102", 1)
    stream._socket = SimpleNamespace(close=delayed_close, transport=transport)
    task = asyncio.create_task(stream._close())
    await closing.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert stream._socket is None
    transport.abort.assert_called_once()


def _voice_cfg(tmp_path, **overrides):
    import yaml
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({
        "magic_phrases": ["hey agent"], "listening_chime": False,
        "speaker": {"enabled": True, **overrides},
    }))
    return load_voice_gate_config(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("require_wake", [False, True])
async def test_enrollment_interacts_with_wake_gate_and_preserves_typed_input(tmp_path, require_wake):
    cfg = _voice_cfg(tmp_path, require_wake_phrase=require_wake)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._use_speaker_asr = True
    gate.push_frame = AsyncMock()
    async def speech(pid, text):
        frame = _SpeakerTranscriptionFrame(text=text, user_id=pid, timestamp="now", speaker_id=7)
        frame.transport_source = pid
        frame.pts = 123000
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    await speech("a", "hey agent look around")
    assert not gate.push_frame.called
    await gate.process_frame(_SpeakerEnrollmentFrame("a", "enrolled"), FrameDirection.DOWNSTREAM)
    gate.push_frame.reset_mock()
    await speech("b", "hey agent look around")
    assert not gate.push_frame.called
    await speech("a", "look around")
    queries = [c.args[0] for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]
    assert bool(queries) is (not require_wake)
    gate.push_frame.reset_mock()
    await speech("a", "hey agent look around")
    queries = [c.args[0] for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]
    expected = "look around" if require_wake else "hey agent look around"
    assert len(queries) == 1 and queries[0].text == expected
    assert queries[0].pts_us == 123
    await gate.process_frame(_SpeakerEnrollmentFrame("a", "released"), FrameDirection.DOWNSTREAM)
    gate.push_frame.reset_mock()
    await speech("a", "hey agent look around")
    assert not gate.push_frame.called
    typed = TranscriptionFrame(text="hey agent typed query", user_id="a", timestamp="now")
    await gate.process_frame(typed, FrameDirection.DOWNSTREAM)
    assert any(isinstance(c.args[0], GatedQueryFrame) for c in gate.push_frame.call_args_list)


@pytest.mark.asyncio
async def test_early_stop_never_runs_for_unenrolled_voice_or_control_prefix():
    processor = _SpeakerSttProcessor(cfg=_SpeakerConfig(), on_partial_transcript=AsyncMock(return_value=False))
    processor.push_frame = AsyncMock()
    await processor._event("a", {"kind": "partial", "text": "stop"})
    assert not processor.push_frame.called
    await processor._event("a", {"kind": "enrolled"})
    processor.push_frame.reset_mock()
    await processor._event("a", {"kind": "partial", "text": "Hey agent let's stop"})
    assert not processor.push_frame.called
    await processor._event("a", {"kind": "partial", "text": "stop"})
    assert isinstance(processor.push_frame.call_args.args[0], InterruptionFrame)
    assert processor._enrolled == {"a"}
    processor.push_frame.reset_mock()
    await processor._event("a", {"kind": "partial", "text": "stop"})
    assert not processor.push_frame.called


@pytest.mark.asyncio
async def test_model_failure_revokes_enrollment_without_batch_stt_fallback():
    processor = _SpeakerSttProcessor(cfg=_SpeakerConfig())
    processor._client = SimpleNamespace(_feed=AsyncMock(side_effect=TimeoutError()), _forget=AsyncMock())
    processor.push_frame = AsyncMock()
    processor._enrolled.add("a")
    audio = InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1)
    audio.transport_source = "a"
    await processor.process_frame(audio, FrameDirection.DOWNSTREAM)
    await processor._tasks["a"]
    assert not processor._enrolled
    assert isinstance(processor.push_frame.call_args.args[0], _SpeakerEnrollmentFrame)
    await processor.process_frame(audio, FrameDirection.DOWNSTREAM)
    processor._client._feed.assert_awaited_once()
    processor.push_frame.reset_mock()
    await processor.process_frame(ParticipantLeftFrame("a"), FrameDirection.DOWNSTREAM)
    processor._client._forget.assert_awaited_once_with("a")


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["cancel", "cleanup"])
async def test_slow_inference_does_not_block_other_participants_and_overload_drops_audio(shutdown):
    processor = _SpeakerSttProcessor(cfg=_SpeakerConfig())
    slow = asyncio.Event()
    other = asyncio.Event()

    async def feed(pid, audio, pts_us):
        if pid == "slow":
            slow.set()
            await asyncio.Future()
        other.set()
        return [{"kind": "enrolled"}]

    processor._client = SimpleNamespace(_feed=AsyncMock(side_effect=feed), _forget=AsyncMock(), _close=AsyncMock())
    processor.push_frame = AsyncMock()
    processor._enrolled.add("slow")
    def audio(pid):
        frame = InputAudioRawFrame(audio=bytes(32000), sample_rate=16000, num_channels=1)
        frame.transport_source = pid
        return frame
    await processor.process_frame(audio("slow"), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(slow.wait(), timeout=1)
    await processor.process_frame(audio("other"), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(other.wait(), timeout=1)
    for _ in range(3):
        await processor.process_frame(audio("slow"), FrameDirection.DOWNSTREAM)
    assert "slow" not in processor._enrolled
    assert "slow" not in processor._queues
    assert "slow" not in processor._tasks
    resets = [c.args[0] for c in processor.push_frame.call_args_list if isinstance(c.args[0], _SpeakerEnrollmentFrame)]
    assert any(f.participant_id == "slow" and f.state == "reset" for f in resets)
    # A reply from a cancelled request cannot restore enrollment.
    assert processor._client._feed.await_count == 2
    if shutdown == "cancel":
        await processor.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
    else:
        await processor.cleanup()
    assert not processor._tasks and not processor._queues
    processor._client._forget.assert_any_await("slow")
    processor._client._forget.assert_any_await("other")
    await processor.process_frame(audio("other"), FrameDirection.DOWNSTREAM)
    assert not processor._tasks


@pytest.mark.asyncio
async def test_stream_isolates_participants_and_reconnects_with_new_origin():
    sessions = {}
    closed = asyncio.Queue()

    async def handler(socket):
        opening = json.loads(await socket.recv())
        assert socket.request.path == "/v1/audio/transcriptions/stream"
        sessions[socket] = opening["audio_origin_us"]
        await socket.send(json.dumps({"service": "xr-ai-speaker-stt", "protocol": 3}))
        count, samples = 0, 0
        try:
            async for audio in socket:
                assert isinstance(audio, bytes) and audio == bytes(640)
                count += 1
                pts_us = opening["audio_origin_us"] + samples * 1_000_000 // 16000
                samples += len(audio) // 2
                events = [{"kind": "transcript", "text": str(count), "pts_us": pts_us}]
                await socket.send(json.dumps({"events": events}))
        finally:
            sessions.pop(socket)
            closed.put_nowait(True)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = _SpeakerClient(_SpeakerConfig(base_url=f"http://127.0.0.1:{port}", timeout_s=1))
        try:
            assert (await client._feed("a", bytes(640), 1))[0]["text"] == "1"
            assert (await client._feed("b", bytes(640), 2))[0]["text"] == "1"
            event = (await client._feed("a", bytes(640), 20_001))[0]
            assert event["text"] == "2" and event["pts_us"] == 20_001
            assert len(sessions) == 2
            await client._forget("a")
            await asyncio.wait_for(closed.get(), timeout=1)
            assert len(sessions) == 1
            event = (await client._feed("a", bytes(640), 1_000_000))[0]
            assert event["text"] == "1" and event["pts_us"] == 1_000_000
        finally:
            await client._close()
            while sessions:
                await asyncio.wait_for(closed.get(), timeout=1)
        assert not client._sessions


@pytest.mark.asyncio
async def test_voice_client_uses_actual_service_health_and_stream_contract():
    closed = asyncio.Queue()

    class Server(_Server):
        def _close(self, session_id):
            super()._close(session_id)
            closed.put_nowait(True)

    class Session:
        def __init__(self, audio_origin_us):
            self.origin = audio_origin_us
            self.samples = 0

        def _feed(self, audio):
            pts_us = self.origin + self.samples * 1_000_000 // 16000
            self.samples += len(audio) // 2
            return [{"kind": "transcript", "text": str(self.samples), "pts_us": pts_us}]

    models = SimpleNamespace(_session=lambda cfg, *, audio_origin_us: Session(audio_origin_us))
    service = Server(models)
    cfg = {"host": "127.0.0.1", "port": 8102, "asr_model": "asr", "diar_model": "diar"}
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    http = uvicorn.Server(uvicorn.Config(_build_app(service, cfg), log_level="error", ws="websockets-sansio"))
    task = asyncio.create_task(http.serve(sockets=[listener]))
    client = _SpeakerClient(_SpeakerConfig(base_url=f"http://127.0.0.1:{port}", timeout_s=1))
    try:
        async with asyncio.timeout(2):
            while not http.started:
                await asyncio.sleep(0)
        assert await client._available()
        assert (await client._feed("wearer", bytes(640), 1_000_000))[0]["pts_us"] == 1_000_000
        assert (await client._feed("wearer", bytes(640), 1_020_000))[0]["pts_us"] == 1_020_000
        assert len(service.sessions) == 1
        await client._forget("wearer")
        await asyncio.wait_for(closed.get(), timeout=1)
        assert not service.sessions
        assert (await client._feed("wearer", bytes(640), 5_000_000))[0]["pts_us"] == 5_000_000
    finally:
        await client._close()
        http.should_exit = True
        await asyncio.wait_for(task, timeout=2)
        listener.close()


@pytest.mark.asyncio
async def test_web_audio_queue_allows_two_seconds_before_reset():
    processor = _SpeakerSttProcessor(cfg=_SpeakerConfig())
    started = asyncio.Event()
    async def feed(*_args):
        started.set()
        await asyncio.Future()
    processor._client = SimpleNamespace(_feed=feed, _forget=AsyncMock(), _close=AsyncMock())
    processor.push_frame = AsyncMock()
    processor._enrolled.add("web-client")
    def audio():
        frame = InputAudioRawFrame(audio=bytes(320), sample_rate=16000, num_channels=1)
        frame.transport_source = "web-client"
        return frame
    try:
        await processor.process_frame(audio(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(started.wait(), timeout=1)
        for _ in range(200):
            await processor.process_frame(audio(), FrameDirection.DOWNSTREAM)
        assert processor._queued_bytes["web-client"] == 64000
        assert "web-client" in processor._enrolled
        await processor.process_frame(audio(), FrameDirection.DOWNSTREAM)
        assert "web-client" not in processor._enrolled
        assert "web-client" not in processor._tasks
    finally:
        await processor.cleanup()


def _config(tmp_path, *, require_wake=False, conversation=True, speaker=True):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({
        "magic_phrases": ["hey agent"], "listening_chime": False,
        "conversation": {"enabled": conversation, "require_wake_phrase": require_wake,
                         "start_phrase": "Hey agent, let's start talking", "phrase_window_s": 6},
        "speaker": {"enabled": speaker, "backend": "auto"},
    }))
    return load_voice_gate_config(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_speaker", [False, True])
@pytest.mark.parametrize("require_wake", [False, True])
async def test_same_conversation_workflow_through_both_asr_paths(tmp_path, monkeypatch, use_speaker, require_wake):
    cfg = _config(tmp_path, require_wake=require_wake)
    stt, captions, accepted = SimpleNamespace(transcribe=AsyncMock()), AsyncMock(), []

    class Detector:
        def __init__(self, on_utterance, on_speech_start, **_kwargs):
            self.start, self.utterance = on_speech_start, on_utterance
        async def feed(self, pcm, rate):
            await self.start()
            await self.utterance(pcm, rate)
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", Detector)

    async def accept(query):
        accepted.append(query)
    transport = HubVoiceTransport()
    output = _VoiceIOProcessor(accept)
    # This harness delivers frames directly instead of running Pipecat queues.
    output._enable_direct_mode = True
    output.push_frame = AsyncMock()
    pipeline, _worker = _build_voice_pipeline(
        transport=transport, stt=stt, tts=SimpleNamespace(), io_processor=output,
        vad_cfg=VadConfig(stop_probe_after_s=0), voice_gate_cfg=cfg,
        on_final_transcript=captions, use_speaker_asr=use_speaker,
    )
    audio, gate = pipeline.processors[2:4]
    assert isinstance(audio, _SpeakerSttProcessor if use_speaker else VadSttProcessor)

    async def to_gate(frame, *_args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    async def to_output(frame, *_args):
        await output.process_frame(frame, FrameDirection.DOWNSTREAM)
    audio.push_frame, gate.push_frame = to_gate, to_output
    selection = _Selection(cfg._speaker)
    now = 0

    async def speak(text):
        nonlocal now
        now += 1_000_000
        if use_speaker:
            selection._activity({7}, 0.3, now)
            for event in selection._finish(text):
                await audio._event("a", event)
        else:
            stt.transcribe.return_value = text
            frame = InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1)
            frame.transport_source, frame.pts = "a", now * 1_000
            await audio.process_frame(frame, FrameDirection.DOWNSTREAM)
        if output._input_tasks:
            await asyncio.gather(*output._input_tasks)

    try:
        await speak("ambient conversation")
        assert not accepted and captions.await_count == 0
        output.push_frame.assert_not_awaited()
        assert not await gate.handle_partial_transcript("a", "stop")
        for part in ("hey agent", "let us", "start talking"):
            await speak(part)
        assert "a" in gate._conversation_active
        assert not accepted and captions.await_count == 0
        assert not await gate.handle_partial_transcript("a", "hey agent let's stop")
        assert await gate.handle_partial_transcript("a", "stop")

        await speak("what is that")
        assert len(accepted) == (0 if require_wake else 1)
        # A wake phrase split from an ordinary command retains its follow-up window.
        await speak("hey agent")
        await speak("describe this")
        assert accepted[-1].text == "describe this"
        count = len(accepted)
        await speak("hey agent")
        await speak("let's stop talking")
        assert "a" not in gate._conversation_active and len(accepted) == count
        await speak("do not send this")
        assert len(accepted) == count
        assert [call.args[1] for call in captions.await_args_list] == ["what is that", "describe this"]
        if use_speaker:
            stt.transcribe.assert_not_awaited()
    finally:
        await audio.cleanup()
        await gate.cleanup()
        await output.cleanup()
        transport.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["auto", "required"])
@pytest.mark.parametrize("available", [False, True])
async def test_backend_is_selected_once_at_startup(monkeypatch, backend, available):
    probe = AsyncMock(return_value=available)
    monkeypatch.setattr(_SpeakerClient, "_available", probe)
    cfg = _SpeakerConfig(backend=backend)
    if backend == "required" and not available:
        with pytest.raises(RuntimeError, match="required but unavailable"):
            await _select_speaker_asr(cfg)
    else:
        assert await _select_speaker_asr(cfg) is available
    assert probe.await_count == 1
    assert await _select_speaker_asr(None) is False
    assert probe.await_count == 1


@pytest.mark.asyncio
async def test_missing_or_incompatible_speaker_endpoint(monkeypatch):
    client = _SpeakerClient(_SpeakerConfig())
    get = AsyncMock(side_effect=httpx.ConnectError("absent"))
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    assert not await client._available()
    get.side_effect = None
    get.return_value = httpx.Response(200, json={"service": "unrelated", "protocol": 3}, request=httpx.Request("GET", "http://localhost/health"))
    with pytest.raises(RuntimeError, match="incompatible service"):
        await client._available()
    get.side_effect = httpx.ReadTimeout("unresponsive")
    assert not await client._available()


@pytest.mark.parametrize("settings", [
    {"backend": "fallback-after-failure"}, {"timeout_s": 0},
    {"timeout_s": True}, {"timeout_s": float("nan")},
])
def test_invalid_backend_configuration_fails_early(settings):
    with pytest.raises(ValueError):
        _SpeakerConfig._from_yaml({"enabled": True, **settings})


def test_conversation_owns_controls_when_both_mappings_are_present(tmp_path):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({
        "conversation": {"enabled": True, "start_phrase": "begin our session", "require_wake_phrase": True},
        "speaker": {"enabled": True, "start_phrase": "ignore this old phrase", "require_wake_phrase": False},
    }))
    cfg = load_voice_gate_config(path)
    assert cfg._speaker.start_phrase == cfg._conversation.start_phrase == "begin our session"
    assert cfg._speaker.require_wake_phrase


def test_conversation_can_use_ordinary_stt_or_preserve_legacy_wake_mode(tmp_path):
    cfg = _config(tmp_path, speaker=False)
    assert cfg._conversation is not None and cfg._speaker is None
    assert not cfg._conversation.require_wake_phrase
    cfg = _config(tmp_path, conversation=False)
    assert cfg._conversation is None and cfg._speaker is None
    assert cfg.magic_phrases == ("hey agent",)
