# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interaction authorization is independent of recognition and enrollment."""

import asyncio
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import uvicorn
import yaml
from pipecat.frames.frames import TranscriptionFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from speaker_stt.__main__ import _build_app, _Server
from speaker_stt._selection import _Selection
from websockets.asyncio.server import serve
from xr_ai_models._speaker_stream import _SpeakerStream
from xr_ai_voice._frames import GatedQueryFrame, ParticipantLeftFrame
from xr_ai_voice._pipeline import _build_voice_pipeline
from xr_ai_voice._processors.io import _VoiceIOProcessor
from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor
from xr_ai_voice._processors.vad_stt import VadConfig
from xr_ai_voice._processors.voice_gate import VoiceGateProcessor
from xr_ai_voice._speaker_client import _select_speaker_asr
from xr_ai_voice._transport import HubVoiceTransport
from xr_ai_voicegate import load_voice_gate_config


def _config(tmp_path, mode, *, speaker=True, stop_enabled=True):
    path = tmp_path / "voice_gate.yaml"
    path.write_text(yaml.safe_dump({
        "mode": mode, "magic_phrases": ["hey agent"], "listening_chime": False,
        "speaker": {"enabled": speaker, "backend": "required"}, "stop_commands_enabled": stop_enabled,
    }))
    return load_voice_gate_config(path)


@pytest.mark.parametrize("mode", ["always_on", "wake_only", "conversation", "hybrid"])
def test_mode_does_not_choose_or_disable_asr_backend(tmp_path, mode):
    cfg = _config(tmp_path, mode)
    assert cfg._mode == mode and cfg._speaker.mode == mode
    assert cfg._speaker.backend == "required"
    assert (cfg._conversation is not None) is (mode in {"conversation", "hybrid"})


@pytest.mark.parametrize("mode", ["wake_only", "hybrid"])
def test_addressed_modes_require_an_actual_phrase(tmp_path, mode):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({"mode": mode}))
    with pytest.raises(ValueError, match="requires non-empty magic_phrases"):
        load_voice_gate_config(path)


@pytest.mark.parametrize("mode", [None, True, 1, "unknown", [], {}])
def test_invalid_mode_never_falls_through_to_an_activation_policy(tmp_path, mode):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({"mode": mode}))
    with pytest.raises(ValueError, match="mode must be"):
        load_voice_gate_config(path)


@pytest.mark.parametrize("mode", ["wake_only", "hybrid"])
def test_attributed_turn_never_enrolls_bystander_or_question_speaker(tmp_path, mode):
    selection = _Selection(_config(tmp_path, mode)._speaker)
    events = selection._enroll_completed([
        (1, "unrelated background conversation", 0, False),
        (7, "hey agent describe the room", 100, False),
    ], completed_at_us=1000)
    assert events == [{"kind": "one_shot", "text": "describe the room", "pts_us": 100,
                       "speaker_id": 7, "fresh_match": True}]
    assert selection.owner is None
    assert selection._enroll_completed([(7, "plain followup", 2000, False)], completed_at_us=3000) == []


@pytest.mark.parametrize("order", [(1, 7), (7, 1)])
def test_same_boundary_one_turn_ambiguity_never_chooses_a_label(tmp_path, order):
    selection = _Selection(_config(tmp_path, "hybrid")._speaker)
    assert selection._enroll_completed([
        (speaker, "hey agent describe this", 0, False) for speaker in order
    ], completed_at_us=100) == []
    assert selection.owner is None
    assert selection._enroll_completed([
        (speaker, selection.cfg.start_phrase, 1000, False) for speaker in order
    ], completed_at_us=2000) == []
    assert selection.owner is None


def test_same_speaker_followup_uses_onset_not_completion_and_is_consumed(tmp_path):
    selection = _Selection(_config(tmp_path, "hybrid")._speaker)
    assert selection._enroll_completed([(7, "hey agent", 0, False)], completed_at_us=1000)[0]["kind"] == "wake_ready"
    assert selection._enroll_completed([(1, "bystander query", 2000, False)], completed_at_us=3000) == []
    # Existing five-second grace: question starts inside, finishes after it.
    events = selection._enroll_completed([(7, "the question", 4_000_000, False)], completed_at_us=9_000_000)
    assert events[0]["text"] == "the question" and events[0]["fresh_match"] is False
    assert selection.owner is None
    assert selection._enroll_completed([(7, "second query", 10_000_000, False)], completed_at_us=11_000_000) == []
    selection._enroll_completed([(7, "hey agent", 12_000_000, False)], completed_at_us=13_000_000)
    assert selection._enroll_completed([(7, "expired", 19_000_000, False)], completed_at_us=20_000_000) == []


@pytest.mark.parametrize("grace,onset", [(0, 1000), (5, 5_001_000)])
def test_followup_disabled_or_at_expiry_never_authorizes(tmp_path, grace, onset):
    from dataclasses import replace

    selection = _Selection(replace(_config(tmp_path, "hybrid")._speaker, followup_grace_s=grace))
    selection._enroll_completed([(7, "hey agent", 0, False)], completed_at_us=1000)
    assert selection._enroll_completed([(7, "not addressed", onset, False)], completed_at_us=onset + 100) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wake_only", "hybrid", "always_on"])
async def test_complete_speaker_turn_acknowledges_wake_once(tmp_path, mode):
    from xr_ai_voice._frames import _SpeakerTurnFrame

    gate = VoiceGateProcessor(cfg=_config(tmp_path, mode), tts=SimpleNamespace())
    gate.push_frame = AsyncMock()
    gate._emit_chime = AsyncMock()
    await gate.process_frame(_SpeakerTurnFrame(
        participant_id="a", speaker_id=7, text="question", pts_us=1000, fresh_match=True,
    ), FrameDirection.DOWNSTREAM)
    assert gate._emit_chime.await_count == (0 if mode == "always_on" else 1)
    assert len([c for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]) == 1


def test_partial_start_control_is_not_a_turn_and_unique_full_start_owns(tmp_path):
    selection = _Selection(_config(tmp_path, "hybrid")._speaker)
    assert selection._enroll_completed([(7, "hey agent let's", 0, False)], completed_at_us=100) == []
    assert selection.owner is None
    events = selection._enroll_completed([
        (1, "hey agent a different question", 1000, False),
        (7, "start talking", 1000, False),
    ], completed_at_us=2000)
    assert events == [{"kind": "enrolled"}] and selection.owner == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("use_speaker", [False, True])
@pytest.mark.parametrize("mode", ["always_on", "wake_only", "conversation", "hybrid"])
@pytest.mark.parametrize("stop_enabled", [False, True])
async def test_actual_gate_and_selection_sequences_match_modes(tmp_path, use_speaker, mode, stop_enabled):
    from pipecat.frames.frames import InterruptionFrame
    cfg = _config(tmp_path, mode, speaker=use_speaker, stop_enabled=stop_enabled)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._use_speaker_asr = use_speaker
    gate.push_frame = AsyncMock()
    captions = AsyncMock()
    gate._on_conversation_transcript = captions
    selection = _Selection(cfg._speaker) if use_speaker else None
    audio = _SpeakerSttProcessor(cfg=cfg._speaker) if use_speaker else None
    if audio:
        async def forward(frame, *_args):
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        audio.push_frame = forward
    now = 0

    async def speak(text, speaker=7):
        nonlocal now
        now += 100_000
        gate.push_frame.reset_mock()
        if audio:
            if selection.owner is None:
                events = selection._enroll_completed([(speaker, text, now, False)], completed_at_us=now + 10_000)
            else:
                selected, _, _ = selection._activity({speaker}, 0.1, now)
                events = selection._finish(text) if selected is not None else []
            for event in events:
                await audio._event("a", event)
        else:
            started = UserStartedSpeakingFrame()
            started.transport_source = "a"
            await gate.process_frame(started, FrameDirection.DOWNSTREAM)
            frame = TranscriptionFrame(text=text, user_id="a", timestamp="now")
            frame.transport_source, frame.pts = "a", now * 1000
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        return [call.args[0] for call in gate.push_frame.call_args_list if isinstance(call.args[0], GatedQueryFrame)]

    try:
        query = await speak("hey agent xyz")
        assert bool(query) is (mode != "conversation")
        assert "a" not in gate._conversation_active
        assert not selection or selection.owner is None
        if query:
            assert query[0].pts_us == now
        query = await speak("unaddressed followup")
        assert bool(query) is (mode == "always_on")
        stopped = await speak("stop")
        assert bool(stopped) is (mode == "always_on" and not stop_enabled)
        interruptions = [call for call in gate.push_frame.call_args_list if isinstance(call.args[0], InterruptionFrame)]
        assert bool(interruptions) is (stop_enabled and mode in {"always_on", "wake_only"})
        if mode in {"wake_only", "hybrid"}:
            assert not await speak("hey agent")
            assert (await speak("one next utterance"))[0].text == "one next utterance"
            assert not await speak("another utterance")
        if mode == "hybrid":
            # Withheld control fragments are judged together, so a split wake
            # phrase still grants the next utterance.
            assert not await speak("hey")
            assert not await speak("agent")
            assert (await speak("one next utterance"))[0].text == "one next utterance"
        if mode in {"conversation", "hybrid"}:
            assert not await speak("hey agent let's")
            assert not await speak("start talking")
            assert "a" in gate._conversation_active
            if selection:
                assert selection.owner == 7
                assert not await speak("hey agent bystander question", speaker=1)
            assert (await speak("natural speech"))[0].text == "natural speech"
            assert not await speak(cfg._conversation.stop_phrase)
            assert "a" not in gate._conversation_active
            if selection:
                assert selection.owner is None
            assert bool(await speak("hey agent after release")) is (mode == "hybrid")
        await gate.process_frame(ParticipantLeftFrame("a"), FrameDirection.DOWNSTREAM)
        assert "a" not in gate._conversation_active
    finally:
        await gate.cleanup()
        if audio:
            await audio.cleanup()


@pytest.mark.asyncio
async def test_always_on_overlay_ignores_existing_wake_phrases_consistently(tmp_path):
    cfg = _config(tmp_path, "always_on", speaker=False)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._emit_text_response = AsyncMock()
    try:
        await gate._on_gate_participant_joined("a")
        gate._emit_text_response.assert_not_awaited()
        assert await gate.handle_partial_transcript("a", "stop")
    finally:
        await gate.cleanup()


@pytest.mark.asyncio
async def test_protocol3_peer_is_rejected_without_audio_or_reuse():
    observed = []
    async def peer(socket):
        observed.append(json.loads(await socket.recv()))
        await socket.send(json.dumps({"service": "xr-ai-speaker-stt", "protocol": 3}))
        await socket.wait_closed()
    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stream = _SpeakerStream(f"http://127.0.0.1:{port}", 1)
        try:
            with pytest.raises(RuntimeError, match="incompatible service"):
                await stream._open({"mode": "hybrid", "magic_phrases": ["hey agent"]}, 100)
        finally:
            await stream._close()
    assert len(observed) == 1 and observed[0]["audio_origin_us"] == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["auto", "required"])
async def test_old_service_health_is_not_absence_or_silent_batch_fallback(tmp_path, backend):
    from dataclasses import replace
    requests = []
    async def peer(reader, writer):
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        body = b'{"service":"xr-ai-speaker-stt","protocol":3}'
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                     + b"\r\nConnection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    async with await asyncio.start_server(peer, "127.0.0.1", 0) as server:
        cfg = replace(_config(tmp_path, "hybrid")._speaker,
                      base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", backend=backend)
        with pytest.raises(RuntimeError, match="incompatible service"):
            await _select_speaker_asr(cfg)
    assert len(requests) == 1 and requests[0].startswith(b"GET /health ")


@pytest.mark.parametrize("grace", [float("nan"), float("inf"), -1, True])
def test_effective_grace_policy_rejected_before_backend_allocation(tmp_path, grace):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({"mode": "hybrid", "magic_phrases": ["hey agent"],
                                    "followup_grace_s": grace, "speaker": {"enabled": True}}))
    with pytest.raises(ValueError, match="finite and non-negative"):
        load_voice_gate_config(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("leaving", ["departure", "shutdown"])
async def test_late_one_turn_cannot_cross_departure_or_shutdown(tmp_path, leaving):
    from pipecat.frames.frames import InputAudioRawFrame
    from xr_ai_voice._frames import ParticipantJoinedFrame, _SpeakerTurnFrame
    cfg = _config(tmp_path, "hybrid")
    processor = _SpeakerSttProcessor(cfg=cfg._speaker)
    entered, release = asyncio.Event(), asyncio.Event()
    async def feed(*_args):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()  # Simulate a request already completing at cancellation.
        return [{"kind": "one_shot", "speaker_id": 7, "text": "late command",
                 "pts_us": 100, "fresh_match": True}]
    processor._client = SimpleNamespace(_feed=feed, _forget=AsyncMock(), _close=AsyncMock())
    processor.push_frame = AsyncMock()
    await processor.process_frame(ParticipantJoinedFrame("a"), FrameDirection.DOWNSTREAM)
    frame = InputAudioRawFrame(audio=bytes(320), sample_rate=16000, num_channels=1)
    frame.transport_source = "a"
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    await entered.wait()
    teardown = asyncio.create_task(
        processor.process_frame(ParticipantLeftFrame("a"), FrameDirection.DOWNSTREAM)
        if leaving == "departure" else processor.cleanup()
    )
    # Cancellation delivery is the deterministic barrier; the old worker must
    # remain owned until the delayed response/cleanup actually completes.
    async with asyncio.timeout(2):
        while "a" in processor._participants:
            await asyncio.sleep(0)
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    release.set()
    await teardown
    assert not [call for call in processor.push_frame.call_args_list if isinstance(call.args[0], _SpeakerTurnFrame)]
    assert not processor._tasks and not processor._queues and not processor._enrolled
    await processor.cleanup()


@pytest.mark.asyncio
async def test_real_speaker_producer_websocket_hybrid_to_runtime_without_enrollment(
    tmp_path, _streaming_models, monkeypatch,
):
    from speaker_stt._inference import _Session
    from xr_ai_voice._frames import ParticipantJoinedFrame

    cfg = _config(tmp_path, "hybrid")
    models, activity, _calls, _masks, _allocations, text = _streaming_models
    recognized = {1: "unrelated continuous background speech", 7: "hey agent describe this"}
    original_step = _Session._step_native

    def step(self, *args):
        text[0] = recognized[args[3]]
        return original_step(self, *args)
    monkeypatch.setattr(_Session, "_step_native", step)
    allocated = []

    def allocate(config, *, audio_origin_us):
        session = _Session(models, config, audio_origin_us=audio_origin_us)
        allocated.append(session)
        return session
    server = _Server(SimpleNamespace(_session=allocate))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    http = uvicorn.Server(uvicorn.Config(_build_app(server, {
        "host": "127.0.0.1", "port": port, "asr_model": "synthetic", "diar_model": "synthetic",
    }), log_level="error", ws="websockets-sansio"))
    serving = asyncio.create_task(http.serve(sockets=[listener]))
    cfg = _config(tmp_path, "hybrid")
    from dataclasses import replace
    object.__setattr__(cfg, "_speaker", replace(
        cfg._speaker, base_url=f"http://127.0.0.1:{port}",
    ))
    accepted, captions, acknowledgements = [], AsyncMock(), asyncio.Queue()
    processed = asyncio.Queue()
    async def accept(query):
        accepted.append(query)
    transport = HubVoiceTransport()
    output = _VoiceIOProcessor(accept)
    output._enable_direct_mode = True
    output.push_frame = AsyncMock()
    pipeline, _worker = _build_voice_pipeline(
        transport=transport, stt=None, tts=SimpleNamespace(), io_processor=output,
        vad_cfg=VadConfig(silence_duration=0.3, stop_probe_after_s=0), voice_gate_cfg=cfg,
        on_final_transcript=captions, use_speaker_asr=True,
    )
    audio, gate = pipeline.processors[2:4]
    async def forward(frame, *_args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    async def dispatch(frame, *_args):
        await output.process_frame(frame, FrameDirection.DOWNSTREAM)
    audio.push_frame, gate.push_frame = forward, dispatch
    original_feed = audio._client._feed
    async def fed(*args):
        events = await original_feed(*args)
        await acknowledgements.put(events)
        return events
    audio._client._feed = fed
    original_event = audio._event
    async def applied(pid, event):
        await original_event(pid, event)
        processed.put_nowait(event)
    audio._event = applied
    now = 1_000_000
    async def feed(speakers):
        nonlocal now
        from pipecat.frames.frames import InputAudioRawFrame
        activity[:] = speakers
        frame = InputAudioRawFrame(audio=bytes(5120), sample_rate=16000, num_channels=1)
        frame.transport_source, frame.pts = "a", now * 1000
        now += 160_000
        await audio.process_frame(frame, FrameDirection.DOWNSTREAM)
        async with asyncio.timeout(2):
            events = await acknowledgements.get()
            for event in events:
                assert await processed.get() == event
        if output._input_tasks:
            await asyncio.gather(*output._input_tasks)
        return events
    try:
        async with asyncio.timeout(2):
            while not http.started:
                await asyncio.sleep(0)
        await audio.process_frame(ParticipantJoinedFrame("a"), FrameDirection.DOWNSTREAM)
        await feed([1])
        await feed([1, 7])
        await feed([1])
        events = await feed([1])
        assert any(event.get("speaker_id") == 7 and event["kind"] == "one_shot" for event in events)
        assert [query.text for query in accepted] == ["describe this"]
        assert allocated[0].selection.owner is None
        assert not audio._enrolled and not gate._conversation_active
        assert [call.args[1] for call in captions.await_args_list] == ["describe this"]
        recognized[7] = "plain followup"
        await feed([1, 7])
        await feed([1])
        await feed([1])
        assert len(accepted) == 1
        recognized[7] = cfg._conversation.start_phrase
        await feed([1, 7])
        await feed([1])
        await feed([1])
        assert allocated[0].selection.owner == 7 and "a" in gate._conversation_active
        recognized[7] = "natural speech"
        await feed([1, 7])
        await feed([1])
        await feed([1])
        assert [query.text for query in accepted] == ["describe this", "natural speech"]
        recognized[7] = cfg._conversation.stop_phrase
        await feed([1, 7])
        await feed([1])
        await feed([1])
        assert allocated[0].selection.owner is None and not gate._conversation_active
        await audio.process_frame(ParticipantLeftFrame("a"), FrameDirection.DOWNSTREAM)
        assert not audio._tasks and not audio._queues and not audio._enrolled
    finally:
        await audio.cleanup()
        await gate.cleanup()
        await output.cleanup()
        transport.shutdown()
        http.should_exit = True
        await asyncio.wait_for(serving, 2)
        listener.close()
    assert not server.sessions
