# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private speaker transport and lifecycle boundary regressions without models."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import DataFrame, InterruptionFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from test_speaker_voice import _voice_cfg
from websockets.asyncio.server import serve
from xr_ai_models._speaker_stream import _speaker_available, _SpeakerStream
from xr_ai_voice import _speaker_client
from xr_ai_voice._frames import (
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
    _SpeakerEnrollmentFrame,
    _TrackedInputAudioFrame,
)
from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor
from xr_ai_voice._processors.voice_gate import VoiceGateProcessor
from xr_ai_voice._speaker_client import _select_speaker_asr, _SpeakerClient
from xr_ai_voicegate._speaker import _SpeakerConfig


def audio(track="one", pts_us=0):
    frame = _TrackedInputAudioFrame(audio=bytes(320), sample_rate=16000, num_channels=1, track_id=track)
    frame.transport_source, frame.pts = "a", pts_us * 1000
    return frame


@pytest.mark.asyncio
async def test_departure_closes_admission_through_blocked_cleanup_and_rejoin():
    p = _SpeakerSttProcessor(cfg=_SpeakerConfig())
    entered, release = asyncio.Event(), asyncio.Event()
    started = asyncio.Queue()
    async def feed(*args):
        await started.put(args)
        await asyncio.Event().wait()
    async def forget(*args):
        entered.set()
        await release.wait()
    p._client = SimpleNamespace(_feed=feed, _forget=forget, _close=AsyncMock())
    p.push_frame = AsyncMock()
    await p.process_frame(audio(), FrameDirection.DOWNSTREAM)
    assert not p._tasks  # audio is not an implicit participant join
    await p.process_frame(ParticipantJoinedFrame("a"), FrameDirection.DOWNSTREAM)
    await p.process_frame(audio(), FrameDirection.DOWNSTREAM)
    await started.get()
    old = p._tasks["a"]
    leaving = asyncio.create_task(p.process_frame(ParticipantLeftFrame("a"), FrameDirection.DOWNSTREAM))
    await entered.wait()
    await p.process_frame(audio(), FrameDirection.DOWNSTREAM)
    assert p._tasks["a"] is old
    release.set()
    await leaving
    await p.process_frame(audio(), FrameDirection.DOWNSTREAM)
    assert not p._tasks and not p._queues
    await p.process_frame(ParticipantJoinedFrame("a"), FrameDirection.DOWNSTREAM)
    await p.process_frame(audio(), FrameDirection.DOWNSTREAM)
    await started.get()
    replacement = p._tasks["a"]
    assert replacement is not old
    await p._shutdown()
    assert replacement.done() and not p._tasks and not p._queues


@pytest.mark.asyncio
async def test_track_change_reopens_without_join_or_timestamp_gap_heuristic():
    p = _SpeakerSttProcessor(cfg=_SpeakerConfig())
    received = asyncio.Queue()
    async def feed(*args):
        await received.put(args)
        return []
    p._client = SimpleNamespace(_feed=feed, _forget=AsyncMock(), _close=AsyncMock())
    p.push_frame = AsyncMock()
    await p.process_frame(ParticipantJoinedFrame("a"), FrameDirection.DOWNSTREAM)
    await p.process_frame(audio("one", 1_000_000), FrameDirection.DOWNSTREAM)
    assert (await received.get())[2] == 1_000_000
    p._enrolled.add("a")
    await p.process_frame(audio("one", 11_000_000), FrameDirection.DOWNSTREAM)
    await received.get()
    p._client._forget.assert_not_awaited()
    await p.process_frame(audio("two", 12_000_000), FrameDirection.DOWNSTREAM)
    assert (await received.get())[2] == 12_000_000
    p._client._forget.assert_awaited_once_with("a")
    assert not p._enrolled
    resets = [c.args[0] for c in p.push_frame.await_args_list if isinstance(c.args[0], _SpeakerEnrollmentFrame)]
    assert [f.state for f in resets] == ["reset"]
    await p.cleanup()


@pytest.mark.asyncio
async def test_real_hub_audio_projection_retains_existing_track_identity():
    from unittest.mock import Mock

    from pipecat.transports.base_transport import TransportParams
    from xr_ai_hub import AudioChunk
    from xr_ai_voice._transport import DeviceIOHubInputTransport

    transport = DeviceIOHubInputTransport(
        SimpleNamespace(on_audio=Mock(), on_participant=Mock()), TransportParams(),
    )
    transport._started = True
    transport.push_frame = AsyncMock()
    await transport._on_hub_audio(AudioChunk(
        pts_us=123, sample_rate=16000, channels=1, samples=160,
        data=bytes(640), participant_id="a", track_id="microphone-two",
    ))
    frame = transport.push_frame.await_args.args[0]
    assert isinstance(frame, _TrackedInputAudioFrame)
    assert frame.track_id == "microphone-two" and frame.transport_source == "a" and frame.pts == 123000


@pytest.mark.asyncio
async def test_wire_payload_matches_independent_protocol3_service_schema(monkeypatch):
    # This is the service schema shipped in merged #585, independent of the
    # worker's expanded config dataclass. Client policy must never enter it.
    accepted = {"start_phrase", "stop_phrase", "require_wake_phrase", "phrase_window_s",
                "activity_threshold", "silence_duration", "max_utterance_s"}
    async def handler(socket):
        opening = json.loads(await socket.recv())
        assert set(opening["config"]) == accepted
        assert opening["audio_origin_us"] == 100
        await socket.send(json.dumps({"service": "xr-ai-speaker-stt", "protocol": 3}))
        await socket.recv()
        await socket.send(json.dumps({"events": []}))
    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = _SpeakerClient(_SpeakerConfig(base_url=f"http://127.0.0.1:{port}"))
        assert await client._feed("a", bytes(320), 100) == []
        await client._close()


@pytest.mark.asyncio
async def test_http_and_websocket_ignore_ambient_proxy_and_normalize_origin(monkeypatch):
    proxy_requests = []
    async def proxy(reader, writer):
        proxy_requests.append(await reader.read(1024))
        writer.close()
    async def health(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        body = b'{"service":"xr-ai-speaker-stt","protocol":3}'
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                     + b"\r\nConnection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
    async def handler(socket):
        await socket.recv()
        await socket.send(json.dumps({"service": "xr-ai-speaker-stt", "protocol": 3}))
    proxy_server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    health_server = await asyncio.start_server(health, "127.0.0.1", 0)
    async with proxy_server, health_server, serve(handler, "127.0.0.1", 0) as ws:
        proxy_url = f"http://127.0.0.1:{proxy_server.sockets[0].getsockname()[1]}"
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(name, proxy_url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        assert await _speaker_available(f"http://127.0.0.1:{health_server.sockets[0].getsockname()[1]}", 1) is True
        stream = _SpeakerStream(f"HTTP://127.0.0.1:{ws.sockets[0].getsockname()[1]}", 1)
        await stream._open({}, 0)
        await stream._close()
        assert not proxy_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,first", [("required", False), ("auto", None)])
async def test_present_or_required_service_waits_existing_readiness(monkeypatch, backend, first):
    probe = AsyncMock(side_effect=[first, None, True])
    monkeypatch.setattr(_SpeakerClient, "_available", probe)
    async def wait(probes):
        assert not await probes["speaker ASR"]()
        assert await probes["speaker ASR"]()
    waiter = AsyncMock(side_effect=wait)
    monkeypatch.setattr(_speaker_client, "wait_for_services", waiter)
    assert await _select_speaker_asr(_SpeakerConfig(backend=backend)) is True
    waiter.assert_awaited_once()
    assert probe.await_count == 3


@pytest.mark.asyncio
async def test_bad_status_is_not_unavailable_fallback(monkeypatch):
    monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=httpx.Response(
        502, request=httpx.Request("GET", "http://localhost/health"))))
    with pytest.raises(httpx.HTTPStatusError):
        await _select_speaker_asr(_SpeakerConfig())


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["enrolled", "released", "reset"])
async def test_enrollment_transitions_survive_actual_interrupted_gate_queue(tmp_path, state):
    gate = VoiceGateProcessor(cfg=_voice_cfg(tmp_path), tts=SimpleNamespace())
    gate.push_frame = AsyncMock()
    gate._emit_text_response = AsyncMock()
    if state != "enrolled":
        gate._conversation_active.add("a")
    entered, handled, interrupted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = gate.process_frame
    blocker = DataFrame()
    async def process(frame, direction):
        if frame is blocker:
            entered.set()
            await asyncio.Event().wait()
        await original(frame, direction)
        if isinstance(frame, _SpeakerEnrollmentFrame):
            handled.set()
        if isinstance(frame, InterruptionFrame):
            interrupted.set()
    gate.process_frame = process
    await gate.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=None))
    await gate.queue_frame(StartFrame())
    try:
        await gate.queue_frame(blocker)
        await asyncio.wait_for(entered.wait(), 1)
        await gate.queue_frame(_SpeakerEnrollmentFrame("a", state))
        while not gate.has_queued_frame(_SpeakerEnrollmentFrame):
            await asyncio.sleep(0)
        await gate.queue_frame(InterruptionFrame())
        await asyncio.wait_for(interrupted.wait(), 1)
        await asyncio.wait_for(handled.wait(), 1)
        assert ("a" in gate._conversation_active) is (state == "enrolled")
    finally:
        await gate.cleanup()


@pytest.mark.asyncio
async def test_active_reset_interrupts_and_notifies_once(tmp_path):
    gate = VoiceGateProcessor(cfg=_voice_cfg(tmp_path), tts=SimpleNamespace())
    gate.push_frame = AsyncMock()
    gate._emit_text_response = AsyncMock()
    gate._conversation_active.add("a")
    for _ in range(3):
        await gate.process_frame(_SpeakerEnrollmentFrame("a", "reset"), FrameDirection.DOWNSTREAM)
    assert sum(isinstance(c.args[0], InterruptionFrame) for c in gate.push_frame.await_args_list) == 1
    gate._emit_text_response.assert_awaited_once()
    assert "reset" in gate._emit_text_response.await_args.args[1]
    await gate.cleanup()
