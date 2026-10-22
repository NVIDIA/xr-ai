# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics observe mode decisions; only the service authorizes turns."""

import asyncio
import json
import socket
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import uvicorn
import yaml
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from speaker_stt.__main__ import _build_app, _Server
from speaker_stt._inference import _Session
from websockets.asyncio.server import serve
from xr_ai_interaction import UserQuery
from xr_ai_models._speaker_stream import _SpeakerStream
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_voice import VadConfig, VoiceAgent
from xr_ai_voice._frames import ParticipantJoinedFrame, ParticipantLeftFrame
from xr_ai_voice._pipeline import _build_voice_pipeline
from xr_ai_voice._processors.io import _VoiceIOProcessor
from xr_ai_voice._speaker_client import _select_speaker_asr
from xr_ai_voice._transport import HubVoiceTransport
from xr_ai_voicegate import load_voice_gate_config
from xr_ai_voicegate._speaker import _SpeakerConfig

_QUERIES = Topic("test.diagnostic-modes.accepted", UserQuery)


@pytest.mark.asyncio
async def test_protocol4_socket_is_rejected_before_audio():
    observed = []

    async def peer(socket):
        observed.append(json.loads(await socket.recv()))
        await socket.send(json.dumps({"service": "xr-ai-speaker-stt", "protocol": 4}))
        await socket.wait_closed()

    async with serve(peer, "127.0.0.1", 0) as server:
        stream = _SpeakerStream(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", 1)
        try:
            with pytest.raises(RuntimeError, match="incompatible service"):
                await stream._open({"mode": "hybrid", "diagnostics": False}, 100)
        finally:
            await stream._close()
    assert observed == [{"config": {"mode": "hybrid", "diagnostics": False}, "audio_origin_us": 100}]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["auto", "required"])
async def test_protocol4_health_cannot_silently_fall_back(backend):
    requests = []

    async def peer(reader, writer):
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        body = b'{"service":"xr-ai-speaker-stt","protocol":4}'
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                     + b"\r\nConnection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async with await asyncio.start_server(peer, "127.0.0.1", 0) as server:
        cfg = _SpeakerConfig(base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", backend=backend)
        with pytest.raises(RuntimeError, match="incompatible service"):
            await _select_speaker_asr(cfg)
    assert len(requests) == 1 and requests[0].startswith(b"GET /health ")


@pytest.mark.parametrize("mode", ["always_on", "wake_only", "conversation", "hybrid"])
def test_diagnostics_preserve_primary_mode_decisions(_streaming_models, monkeypatch, mode):
    models, activity, _calls, _masks, _allocations, text = _streaming_models

    def trace(diagnostics):
        cfg = _SpeakerConfig(mode=mode, magic_phrases=("hey agent",), diagnostics=diagnostics, silence_duration=0.16)
        session = _Session(models, cfg, audio_origin_us=0)
        recognized, primary, display, states = {}, [], [], []
        original = session._decode

        def decode(*args):
            text[0] = recognized.get(args[3], "")
            return original(*args)
        monkeypatch.setattr(session, "_decode", decode)

        def feed(speakers, words=None):
            activity[:] = speakers
            recognized.update(words or {})
            events = session._feed(bytes(5120))
            primary.extend(event for event in events if event["kind"] != "diagnostic")
            display.extend(event for event in events if event["kind"] == "diagnostic")
            states.append((session.selection.owner, dict(session.selection._wake_until)))

        def utter(speaker, words):
            feed([speaker], {speaker: words})
            feed([])

        # An already-speaking bystander cannot reserve the addressed user's turn.
        feed([1], {1: "ordinary background speech"})
        feed([1, 7], {7: "hey agent first question"})
        feed([1])
        assert session.selection.owner is None
        feed([])
        utter(7, "plain followup")
        utter(7, "hey agent")
        utter(1, "bystander followup")
        utter(7, "same speaker followup")
        # Partial wake/control fragments are not independent addressed turns.
        utter(7, "hey")
        utter(7, "agent")
        utter(7, "unaddressed after fragments")
        feed([2, 7], {2: cfg.start_phrase, 7: cfg.start_phrase})
        feed([])
        assert session.selection.owner is None
        if mode in {"conversation", "hybrid"}:
            utter(7, "hey agent let's")
            utter(7, "start talking")
            assert session.selection.owner == 7
            utter(1, "hey agent bystander takeover")
            utter(7, "natural owner query")
            utter(7, cfg.stop_phrase)
            assert session.selection.owner is None
        utter(7, "stop")
        utter(7, "hey agent after stop")
        if diagnostics:
            assert display
            if mode != "conversation":
                assert not any(event["speaker_id"] == 7 and event["text"] == "hey agent first question"
                               for event in display)
                assert not any(event["speaker_id"] == 7 and event["text"] == "hey agent" for event in display)
        else:
            assert not display
        return primary, states

    assert trace(False) == trace(True)


class _Recorder(Agent):
    def __init__(self):
        super().__init__()
        self.queries = []

    @subscribe(_QUERIES)
    async def record(self, query: UserQuery, ctx: RuntimeContext):
        self.queries.append((ctx.metadata.participant_id, query.text, query.timestamp_us))


@pytest.mark.parametrize("mode", ["always_on", "wake_only", "conversation", "hybrid"])
@pytest.mark.parametrize("diagnostics", [False, True])
async def test_actual_speaker_wire_modes_reach_shared_display_and_runtime(
    tmp_path, _streaming_models, monkeypatch, mode, diagnostics,
):
    models, activity, _calls, _masks, _allocations, text = _streaming_models
    recognized = {1: "unrelated background speech", 7: "hey agent describe this"}
    original_decode = _Session._decode

    def decode(self, *args):
        text[0] = recognized[args[3]]
        return original_decode(self, *args)
    monkeypatch.setattr(_Session, "_decode", decode)
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
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({
        "mode": mode, "magic_phrases": ["hey agent"], "listening_chime": False,
        "speaker": {"enabled": True, "backend": "required", "diagnostics": diagnostics},
    }))
    cfg = load_voice_gate_config(path)
    object.__setattr__(cfg, "_speaker", replace(
        cfg._speaker, base_url=f"http://127.0.0.1:{port}", silence_duration=0.3,
    ))
    runtime, recorder = AgentRuntime(), _Recorder()
    runtime.register("recorder", recorder)
    voice = VoiceAgent(query_topic=_QUERIES, stt=None, tts=SimpleNamespace(), vad=VadConfig(), voice_gate=cfg)
    runtime.register("voice", voice)
    voice._runtime = runtime
    display, captions = AsyncMock(), AsyncMock()
    transport = HubVoiceTransport()
    monkeypatch.setattr(transport, "send_return_data", display)
    output = _VoiceIOProcessor(
        voice._publish_input, transport=transport,
    )
    output._enable_direct_mode = True
    output.push_frame = AsyncMock()
    pipeline, _worker = _build_voice_pipeline(
        transport=transport, stt=None, tts=SimpleNamespace(), io_processor=output,
        vad_cfg=VadConfig(stop_probe_after_s=0), voice_gate_cfg=cfg,
        on_final_transcript=captions, use_speaker_asr=True,
    )
    audio, gate = pipeline.processors[2:4]

    async def forward(frame, *_args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    async def dispatch(frame, *_args):
        await output.process_frame(frame, FrameDirection.DOWNSTREAM)
    audio.push_frame, gate.push_frame = forward, dispatch
    applied, responses = asyncio.Queue(), asyncio.Queue()
    original_feed, original_event = audio._client._feed, audio._event

    async def fed(*args):
        events = await original_feed(*args)
        responses.put_nowait(events)
        return events
    async def event(pid, payload):
        await original_event(pid, payload)
        applied.put_nowait(payload)
    audio._client._feed, audio._event = fed, event
    now = 1_000_000

    async def feed(speakers):
        nonlocal now
        activity[:] = speakers
        frame = InputAudioRawFrame(audio=bytes(5120), sample_rate=16000, num_channels=1)
        frame.transport_source, frame.pts = "a", now * 1000
        now += 160_000
        await audio.process_frame(frame, FrameDirection.DOWNSTREAM)
        async with asyncio.timeout(2):
            events = await responses.get()
            for payload in events:
                assert await applied.get() == payload
        if output._input_tasks:
            await asyncio.gather(*output._input_tasks)

    async def utter(words):
        recognized[7] = words
        await feed([1, 7])
        await feed([1])
        await feed([1])

    try:
        async with runtime:
            async with asyncio.timeout(2):
                while not http.started:
                    await asyncio.sleep(0)
            await audio.process_frame(ParticipantJoinedFrame("a"), FrameDirection.DOWNSTREAM)
            await feed([1])
            await utter("hey agent describe this")
            assert allocated[0].selection.cfg.diagnostics is diagnostics
            assert allocated[0].selection.owner is None and not gate._conversation_active
            accepted = "hey agent describe this" if mode == "always_on" else "describe this"
            assert [q[1] for q in recorder.queries] == ([] if mode == "conversation" else [accepted])
            await utter("plain followup")
            assert len(recorder.queries) == (2 if mode == "always_on" else 0 if mode == "conversation" else 1)
            if mode in {"conversation", "hybrid"}:
                await utter(cfg._conversation.start_phrase)
                assert allocated[0].selection.owner == 7 and "a" in gate._conversation_active
                await utter("natural speech")
                await utter(cfg._conversation.stop_phrase)
                assert allocated[0].selection.owner is None and not gate._conversation_active
            messages = [call.args[0] for call in display.await_args_list]
            accepted_messages = [message for message in messages if message.topic == "voice.user-query"]
            assert not any(message.topic == _QUERIES.name for message in messages)
            assert [(message.participant_id, message.pts_us) for message in accepted_messages] == [
                (q[0], q[2]) for q in recorder.queries
            ]
            assert [message.data.decode() for message in accepted_messages] == [
                f"Speaker 7: {q[1]}" for q in recorder.queries
            ]
            ignored = [message for message in messages if message.topic == "voice.speaker-diagnostic"]
            assert bool(ignored) is diagnostics
            if mode != "conversation":
                assert not any(message.data.decode().endswith(": hey agent describe this") for message in ignored)
            assert [call.args[1] for call in captions.await_args_list] == [q[1] for q in recorder.queries]
            assert gate._feeding_speaker_id is None
            await audio.process_frame(ParticipantLeftFrame("a"), FrameDirection.DOWNSTREAM)
            assert not audio._tasks and not audio._queues and not audio._enrolled
    finally:
        voice._runtime = None
        await audio.cleanup()
        await gate.cleanup()
        await output.cleanup()
        transport.shutdown()
        http.should_exit = True
        await asyncio.wait_for(serving, 2)
        listener.close()
    assert not server.sessions
