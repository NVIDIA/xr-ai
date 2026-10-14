# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-controlled task capture, packet integrity, and launch configuration."""

import asyncio
import importlib.util
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit

import pytest
from PIL import Image
from xr_ai_hub import FrameUnavailable, ParticipantEvent
from xr_ai_hub._capture import CAPTURE_STT_TOPIC
from xr_ai_models import load_models_config
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/task-capture"
sys.path.insert(0, str(_SAMPLE / "worker"))

from task_worker.capture.agent import CAPTURE_INPUT_TOPIC, CaptureAgent  # noqa: E402
from task_worker.capture.captioning import caption_query  # noqa: E402
from task_worker.capture.capturing import _Capture  # noqa: E402
from task_worker.config import load_config  # noqa: E402


def test_config_and_cli(monkeypatch, tmp_path):
    worker = load_config(_SAMPLE / "yaml/worker.yaml")
    assert not hasattr(worker, "media_capture_dir")
    assert (worker.capture.capture_fps, worker.capture.caption_interval_s) == (2, 5)
    load_models_config(worker.models_config)
    spec = importlib.util.spec_from_file_location("sop_main", _SAMPLE / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._parser().parse_args([]).output_root is None
    with pytest.raises(SystemExit) as error:
        module._parser().parse_args(["--capture"])
    assert error.value.code == 2
    assert [p.name for p in module.PROCESSES] == ["hub", "worker"]
    stack = Mock()
    monkeypatch.setattr(module, "run_stack", stack)
    module.run([])
    assert [p.name for p in stack.call_args.args[0]] == ["hub", "worker"]
    assert stack.call_args.args[0][1].config == "yaml/worker.yaml"
    assert stack.call_args.args[0][1].command == "task_capture_worker"

    monkeypatch.chdir(tmp_path)
    requests = []

    def launch(processes, base):
        assert [p.name for p in processes] == ["hub", "worker"]
        assert base == _SAMPLE
        request = processes[1].config
        assert request.is_file()
        requests.append(request)
        config = load_config(request)
        assert config.capture.artifacts_dir == tmp_path / "my evidence"
        assert config.models_config == worker.models_config
        assert config.voice_gate_yaml == worker.voice_gate_yaml
        assert config.capture.caption_prompt == worker.capture.caption_prompt
        assert config.capture.capture_fps == worker.capture.capture_fps

    original_yaml = (_SAMPLE / "yaml/worker.yaml").read_bytes()
    original_capture_yaml = (_SAMPLE / "yaml/capture-agent-config.yaml").read_bytes()
    monkeypatch.setattr(module, "run_stack", launch)
    module.run(["--output-root", "my evidence"])
    module.run(["--output-root", str(tmp_path / "my evidence")])
    assert all(not path.exists() for path in requests)
    assert (_SAMPLE / "yaml/worker.yaml").read_bytes() == original_yaml
    assert (_SAMPLE / "yaml/capture-agent-config.yaml").read_bytes() == original_capture_yaml


@pytest.mark.parametrize("absolute_reference", [False, True])
def test_output_root_defaults_to_yaml_and_preserves_relative_paths(tmp_path, absolute_reference):
    config_path = tmp_path / "settings" / "worker.yaml"
    config_path.parent.mkdir()
    capture_path = tmp_path / "capture-settings" / "capture-agent-config.yaml"
    capture_path.parent.mkdir()
    capture_path.write_text("artifacts_dir: evidence\ncapture_fps: 3\ncaption_interval_s: 8\n")
    reference = capture_path if absolute_reference else "../capture-settings/capture-agent-config.yaml"
    config_path.write_text(f"capture_config: {reference}\nsilence_duration: 0.8\nframe_timeout_s: 4\n")
    config = load_config(config_path)
    assert config.capture.artifacts_dir == capture_path.parent / "evidence"
    assert (config.capture.capture_fps, config.capture.caption_interval_s) == (3, 8)
    assert (config.silence_duration, config.frame_timeout_s) == (0.8, 4)
    assert config.models_config == config_path.parent / "models.json"
    assert config.voice_gate_yaml == config_path.parent / "voice_gate.yaml"
    request = tmp_path / "launch.json"
    request.write_text(json.dumps({"worker_config": str(config_path), "output_root": str(tmp_path / "override")}))
    overridden = load_config(request)
    assert overridden.capture.artifacts_dir == tmp_path / "override"
    assert overridden.capture.capture_fps == config.capture.capture_fps
    assert overridden.capture.caption_prompt == config.capture.caption_prompt
    assert overridden.models_config == config.models_config
    assert overridden.voice_gate_yaml == config.voice_gate_yaml


@pytest.mark.parametrize("name", ["capture_fps", "caption_interval_s"])
@pytest.mark.parametrize("value", ["0", "-1", ".inf", ".nan"])
def test_capture_config_rejects_invalid_intervals(tmp_path, name, value):
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("{}\n")
    (tmp_path / "capture-agent-config.yaml").write_text(f"{name}: {value}\n")
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        load_config(config_path)


@pytest.mark.parametrize("gpu_profile", ["spark", "dual_48G_ada", "96G_blackwell"])
def test_documented_model_deployment_matches_worker(gpu_profile, monkeypatch):
    from xr_ai_voicegate import load_voice_gate_config

    server_main = _SAMPLE.parents[1] / "model-server-samples/model-servers/main.py"
    spec = importlib.util.spec_from_file_location("task_capture_model_servers", server_main)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.chdir(_SAMPLE)
    config = load_config(_SAMPLE / "yaml/worker.yaml")
    models = load_models_config(config.models_config)
    processes, _ = launcher._build_processes(Path("yaml/models.json"), gpu_profile)
    services = {process.name: process for process in processes}
    assert set(services) == {"stt", "tts", "vlm"}
    for name, model in models.entries.items():
        assert urlsplit(model.endpoint.base_url).port == services[name].port
        assert services[name].launch_mode == "persist"
    # Match the batch protocol as well as the port; speaker-STT is not a
    # drop-in replacement and would require an enrollment conversation.
    assert models.stt("stt").kind == "openai_compat"
    assert services["stt"].command == "stt_server"
    assert services["stt"].project == "../../services/stt-server"
    gate = load_voice_gate_config(config.voice_gate_yaml)
    assert gate._speaker is None
    assert gate._conversation is None


@pytest.mark.parametrize("document", [
    _SAMPLE / "README.md",
    _SAMPLE.parents[1] / "docs/source/reference/task-capture.md",
])
def test_documented_model_commands_select_sample_deployment(document):
    commands = [line for line in document.read_text().splitlines()
                if line.startswith("uv run --project ") and " model_servers" in line]
    assert commands == [
        "uv run --project ../../model-server-samples/model-servers model_servers --models yaml/models.json",
        "uv run --project ../../model-server-samples/model-servers model_servers --models yaml/models.json --stop",
    ]


@pytest.mark.parametrize("failure", [None, "one_invalid", "all_invalid", "unavailable"])
def test_caption_eval_cli_reports_contract_failures(tmp_path, monkeypatch, capsys, failure):
    import runpy

    import xr_ai_models
    import yaml

    cases = yaml.safe_load((_SAMPLE / "eval/caption_cases.yaml").read_text())["cases"]
    valid = json.dumps({"activity": "Arrange", "phase": "Place", "caption": "A block", "delta": "Unchanged"})
    replies = [SimpleNamespace(content=valid) for _ in cases]
    if failure == "one_invalid":
        replies[0] = SimpleNamespace(content="not JSON")
    elif failure == "all_invalid":
        replies = [SimpleNamespace(content="not JSON") for _ in cases]
    vlm = Mock(ask_images=AsyncMock(side_effect=(RuntimeError("offline") if failure == "unavailable" else replies)),
               close=AsyncMock())
    monkeypatch.setattr(xr_ai_models, "make_vlm", lambda *args: vlm)
    output = tmp_path / "caption-eval"
    monkeypatch.setattr(sys, "argv", [str(_SAMPLE / "eval/caption.py"), "--output", str(output)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(_SAMPLE / "eval/caption.py"), run_name="__main__")
    assert result.value.code == (0 if failure is None else 1)
    vlm.close.assert_awaited_once()
    assert vlm.ask_images.await_count == len(cases)
    records = [json.loads((output / f'{case["id"]}.json').read_text()) for case in cases]
    expected_passes = len(cases) if failure is None else len(cases) - 1 if failure == "one_invalid" else 0
    assert sum("caption" in record for record in records) == expected_passes
    assert f"Caption contract: {expected_passes}/{len(cases)}" in capsys.readouterr().out
    assert (output / "prompt.txt").is_file()


async def test_capture_voice_gate_does_not_interrupt_or_acknowledge_stop():
    from xr_ai_voicegate import VoiceGate, load_voice_gate_config

    config = load_voice_gate_config(_SAMPLE / "yaml/voice_gate.yaml")
    gate = VoiceGate(config, tts=Mock(), audio_sink=Mock())
    query, stop = AsyncMock(), AsyncMock()
    gate.bind(on_query=query, on_stop=stop)
    for text in ("stop", "stop capture", "start capture", "be quiet"):
        assert not gate._matches_partial_stop(text)
        await gate.feed("user", text)
    stop.assert_not_awaited()
    assert len(query.await_args_list) == 4


@pytest.fixture
async def demo(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(image, format="JPEG")
    images = ImageRegistry()
    frames = Mock(execute=AsyncMock(return_value=ImageFrame(
        image=images.put(image.getvalue(), owner="user"), width=8, height=8,
        timestamp_us=time.time_ns() // 1000, sequence=1, participant_id="user", track_id="camera",
    )))
    captioner = Mock(execute=AsyncMock(return_value=SimpleNamespace(
        available=True, text=json.dumps(
            {"activity": "Setup", "phase": "Arrange", "caption": "A blue block", "delta": "Initial view"}
        ),
    )))
    capture_backend = _Capture(
        sessions_dir=tmp_path / "sessions", current_frame=frames, images=images,
        query_images=captioner, capture_fps=2, caption_interval_s=5,
    )
    capture = CaptureAgent(capture_backend)
    try:
        yield SimpleNamespace(
            capture_backend=capture_backend, capture=capture, root=tmp_path, frames=frames,
            captioner=captioner, output=AsyncMock(),
        )
    finally:
        await capture.close()


def joined(pid="user", session="session", pts_us=1):
    return ParticipantEvent(pid, True, pts_us, participant_session_id=session)


async def speak(demo, text, pid="user", timestamp_us=None, wait=True):
    timestamp_us = time.time_ns() // 1000 if timestamp_us is None else timestamp_us
    await demo.capture.on_input(
        UserQuery(text=text, timestamp_us=timestamp_us),
        SimpleNamespace(metadata=SimpleNamespace(participant_id=pid), publish=demo.output),
    )
    if wait and (task := demo.capture._tails.get(pid)):
        await asyncio.shield(task)


def packet(state):
    return json.loads((state.directory / "packet.json").read_text())


def narration(state):
    return [json.loads(line)["text"] for line in (state.directory / "transcript.jsonl").read_text().splitlines()]


def announcements(demo):
    return [call.args[1].text for call in demo.output.await_args_list]


async def test_join_is_idle_and_voice_can_record_multiple_sessions(demo):
    event = joined()
    await demo.capture.receive(event)
    await speak(demo, "Before recording")
    await speak(demo, "stop capture")
    assert not list((demo.root / "sessions").iterdir())
    demo.frames.execute.assert_not_awaited()
    demo.captioner.execute.assert_not_awaited()
    assert announcements(demo) == []
    for _ in range(2):
        await speak(demo, " Start  CAPTURE! ")
        state = demo.capture_backend._sessions["user"]
        await speak(demo, "start capture")  # Duplicate command, not narration or another packet.
        await demo.capture.receive(event)
        assert demo.capture_backend._sessions["user"] is state
        await speak(demo, "Place the cup")
        async with asyncio.timeout(2):
            while state.caption_count == 0:
                await asyncio.sleep(0.001)
        await speak(demo, "Stop capture.")
        assert packet(state)["status"] == packet(state)["narration_status"] == "complete"
        assert packet(state)["counts"] == {"frames": 1, "transcripts": 1, "captions": 1}
        assert packet(state)["stop_command"]["text"] == "Stop capture."
        assert narration(state) == ["Place the cup"]
        await speak(demo, "After recording")
        await demo.capture.receive(event)
        assert not demo.capture_backend.is_capturing("user")
        assert narration(state) == ["Place the cup"]
    assert len(list((demo.root / "sessions").iterdir())) == 2
    assert announcements(demo) == [
        "Capture started.", "Capture stopped.",
    ] * 2
    for call in demo.output.await_args_list:
        assert call.args[0] == VOICE_OUTPUT_TOPIC
        assert call.kwargs["participant_id"] == "user"
        assert call.args[1].interrupt is False


@pytest.mark.parametrize("disconnect", [False, True])
async def test_stop_acknowledgement_precedes_silent_finalization(demo, monkeypatch, disconnect):
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.capture_backend._finalize

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        await original(*args, **kwargs)

    monkeypatch.setattr(demo.capture_backend, "_finalize", delayed)
    await speak(demo, "stop capture", wait=False)
    await asyncio.wait_for(entered.wait(), 2)
    assert announcements(demo) == ["Capture started.", "Capture stopped."]
    assert packet(state)["status"] == "recording"
    departure = None
    try:
        if disconnect:
            departure = asyncio.ensure_future(demo.capture.receive(
                ParticipantEvent("user", False, 2, participant_session_id="session"),
            ))
            await asyncio.sleep(0)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(demo.capture._tails["user"]), 2)
        if departure is not None:
            await departure
    assert packet(state)["status"] == "complete"
    assert announcements(demo) == ["Capture started.", "Capture stopped."]


async def test_announcement_failure_does_not_prevent_finalization(demo):
    demo.output.side_effect = RuntimeError("voice unavailable")
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await speak(demo, "Place the cup")
    await speak(demo, "stop capture")
    assert packet(state)["status"] == "complete"
    assert narration(state) == ["Place the cup"]


async def test_finalization_exception_does_not_add_announcements(demo, monkeypatch):
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    # Fail the requested stop; allow fixture shutdown to drain the real packet.
    with monkeypatch.context() as patch:
        patch.setattr(demo.capture_backend, "finish_capture", AsyncMock(side_effect=OSError("disk full")))
        await speak(demo, "stop capture")
    assert announcements(demo) == [
        "Capture started.", "Capture stopped.",
    ]


async def test_runtime_shutdown_during_stop_announcement_preserves_stop_record(demo):
    from xr_ai_runtime import Agent, RuntimeContext, subscribe
    from xr_ai_voice import VoiceOutput

    entered = asyncio.Event()

    class BlockedOutput(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def output(self, message: VoiceOutput, ctx: RuntimeContext):
            assert message.text == "Capture stopped."
            entered.set()
            await asyncio.Future()

    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await speak(demo, "Place the cup")
    runtime = AgentRuntime()
    runtime.register("capture", demo.capture)
    runtime.register("output", BlockedOutput())
    ts = time.time_ns() // 1000
    try:
        async with runtime:
            await runtime.publish(
                CAPTURE_INPUT_TOPIC, UserQuery(text="Stop capture.", timestamp_us=ts), participant_id="user",
            )
            await asyncio.wait_for(entered.wait(), 2)
            # Use the real runtime cancellation: publication wraps the cancelled
            # output subscriber in BaseExceptionGroup, not ExceptionGroup.
            await runtime.stop()
        await demo.capture.close()
    finally:
        await runtime.stop()
    assert packet(state)["stop_command"] == {"text": "Stop capture.", "timestamp_us": ts}
    assert packet(state)["status"] == "complete"
    assert narration(state) == ["Place the cup"]


async def test_close_drains_other_participants_after_cancelled_action(demo, monkeypatch):
    for pid in ("user", "other"):
        await demo.capture.receive(joined(pid))
        await speak(demo, "start capture", pid)
    state = demo.capture_backend._sessions["other"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.capture_backend.capture_transcript

    async def delayed(*args):
        entered.set()
        await release.wait()
        await original(*args)

    monkeypatch.setattr(demo.capture_backend, "capture_transcript", delayed)
    # A failed participant tail must not short-circuit draining another one.
    failed = demo.capture._enqueue("user", lambda: asyncio.sleep(60))
    failed.cancel()
    await asyncio.gather(failed, return_exceptions=True)
    await speak(demo, "Finish placing the cup", "other", wait=False)
    await asyncio.wait_for(entered.wait(), 2)
    original_stop = demo.capture_backend.stop

    async def stop_after_drain():
        assert demo.capture._tails["other"].done()
        await original_stop()

    monkeypatch.setattr(demo.capture_backend, "stop", stop_after_drain)
    closing = asyncio.create_task(demo.capture.close())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert packet(state)["status"] == "complete"
    assert narration(state) == ["Finish placing the cup"]


async def test_disconnect_and_old_events_cannot_stop_new_recording(demo):
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    old = demo.capture_backend._sessions["user"]
    await speak(demo, "First procedure")
    await demo.capture.receive(ParticipantEvent("user", False, 2, participant_session_id="session"))
    assert packet(old)["status"] == "complete"
    await demo.capture.receive(joined(session="new", pts_us=100))
    assert not demo.capture_backend.is_capturing("user")
    await speak(demo, "start capture", timestamp_us=101)
    await demo.capture.receive(ParticipantEvent("user", False, 3, participant_session_id="session"))
    await speak(demo, "stop capture", timestamp_us=50)
    assert demo.capture_backend.is_capturing("user")

@pytest.mark.parametrize("stop_command", [False, True])
async def test_app_drains_voice_transcripts_and_recordings_before_endpoint_closes(demo, monkeypatch, stop_command):
    from dataclasses import replace

    from task_worker import app
    from xr_ai_voice._session import _VoiceSession
    from xr_ai_voice._types import VoiceQuery

    receiving = asyncio.Event()
    receiver_cancelled = asyncio.Event()
    capture_closed = False
    voice_closed = False
    callbacks = {}

    async def send(message):
        assert not capture_closed
        assert message.topic == CAPTURE_STT_TOPIC

    async def receive_forever():
        receiving.set()
        try:
            await asyncio.Future()
        finally:
            receiver_cancelled.set()

    capture_endpoint = SimpleNamespace(
        on_participant=lambda cb: callbacks.update(participant=cb),
        run=receive_forever,
        wait_until_running=receiving.wait,
        send_return_data=AsyncMock(side_effect=send),
        # ProcessorEndpoint.stop does not wake its blocked ZMQ receive.
        stop=Mock(),
    )

    def close_capture():
        nonlocal capture_closed
        packets = list((demo.root / "app-sessions/sessions").glob("*/packet.json"))
        assert len(packets) == 1
        packet = json.loads(packets[0].read_text())
        assert packet["status"] == packet["narration_status"] == "complete"
        assert packet["counts"]["transcripts"] == 1
        assert voice_closed
        capture_closed = True

    capture_endpoint.close = close_capture
    monkeypatch.setattr(app, "ProcessorEndpoint", lambda **kwargs: capture_endpoint)
    monkeypatch.setattr(app, "ImageRegistry", lambda: demo.capture_backend._images)
    monkeypatch.setattr(app, "CurrentFrameTool", lambda **kwargs: demo.frames)
    monkeypatch.setattr(app, "setup_logging", lambda *args: None)
    for name in ("make_stt", "make_tts", "make_vlm"):
        monkeypatch.setattr(app, name, lambda *args: Mock(health=AsyncMock(return_value=True), close=AsyncMock()))

    def shutdown_voice():
        nonlocal voice_closed
        voice_closed = True

    transport = SimpleNamespace(endpoint=Mock(), shutdown=shutdown_voice, send_return_data=AsyncMock(side_effect=send))
    real_voice = app.VoiceAgent
    voices = []
    spoken = AsyncMock()

    def make_voice(**kwargs):
        assert set(kwargs["probes"]) == {"stt", "tts", "vlm"}
        voice = real_voice(transport=transport, **kwargs)
        monkeypatch.setattr(voice._session, "enqueue_response", spoken)
        voices.append(voice)
        return voice

    monkeypatch.setattr(app, "VoiceAgent", make_voice)

    async def run_session(self, *args, **kwargs):
        assert self.vad.stop_probe_after_s == 0
        async def speech(text):
            ts = time.time_ns() // 1000
            await voices[0]._publish_transcript("user", text, ts)
            await voices[0]._publish_input(VoiceQuery(participant_id="user", text=text, timestamp_us=ts))

        await callbacks["participant"](joined())
        # End the voice run while the participant is still connected.
        # VoiceAgent's real finally path closes only its own transport.
        await speech("Start capture.")
        async with asyncio.timeout(2):
            while not list((demo.root / "app-sessions/sessions").glob("*/packet.json")):
                await asyncio.sleep(0.001)
            while spoken.await_count < 1:
                await asyncio.sleep(0.001)
        await speech("Place a cup")
        # Delivered final transcripts must drain into the packet; speech still
        # awaiting STT at transport shutdown is outside this guarantee.
        await voices[0]._transcript_queue.join()
        if stop_command:
            # Exercise shared transcript publication and accepted command delivery.
            await speech("Stop capture.")
            async with asyncio.timeout(2):
                while True:
                    packets = list((demo.root / "app-sessions/sessions").glob("*/packet.json"))
                    if packets and json.loads(packets[0].read_text())["status"] == "complete":
                        break
                    await asyncio.sleep(0.001)
            packet = json.loads(packets[0].read_text())
            assert packet["counts"]["transcripts"] == 1
            assert packet["stop_command"]["text"] == "Stop capture."
            assert not voice_closed
            async with asyncio.timeout(2):
                while spoken.await_count < 2:
                    await asyncio.sleep(0.001)

    monkeypatch.setattr(_VoiceSession, "run", run_session)
    config = load_config(_SAMPLE / "yaml/worker.yaml")
    config = replace(config, capture=replace(config.capture, artifacts_dir=demo.root / "app-sessions"))
    task = asyncio.create_task(app.run_app(config))
    try:
        done, _ = await asyncio.wait({task}, timeout=2)
        assert task in done, "capture receiver blocked application shutdown"
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert capture_closed
    assert receiver_cancelled.is_set()
    assert spoken.await_args_list[0].args == ("user", "Capture started.")
    if stop_command:
        assert spoken.await_args_list[1].args == ("user", "Capture stopped.")
    assert spoken.await_count == (2 if stop_command else 1)


@pytest.mark.parametrize("raw_transcript_first", [True, False])
async def test_recording_acknowledgements_follow_voice_input(demo, monkeypatch, raw_transcript_first):
    """Exercise both STT delivery paths through real gate, I/O, runtime, and TTS."""
    import wave

    from pipecat.frames.frames import EndFrame, OutputAudioRawFrame, TranscriptionFrame, TTSStoppedFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.workers.runner import WorkerRunner
    from xr_ai_voice import VadConfig, VoiceAgent
    from xr_ai_voice._processors import StreamingTtsProcessor, VoiceGateProcessor, _VoiceIOProcessor
    from xr_ai_voicegate import load_voice_gate_config

    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(640))
    tts = SimpleNamespace(synthesize=AsyncMock(return_value=wav_bytes.getvalue()), close=AsyncMock())
    transport = SimpleNamespace(
        endpoint=Mock(), shutdown=Mock(), send_return_data=AsyncMock(),
    )
    voice = VoiceAgent(
        query_topic=CAPTURE_INPUT_TOPIC, stt=Mock(close=AsyncMock()), tts=tts,
        vad=VadConfig(stop_probe_after_s=0),
        voice_gate=load_voice_gate_config(_SAMPLE / "yaml/voice_gate.yaml"),
        transport=transport, text_input=False,
    )
    boundaries = asyncio.Queue()
    audio_frames = []

    class OutputSink(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, OutputAudioRawFrame):
                audio_frames.append(frame)
            elif isinstance(frame, TTSStoppedFrame):
                boundaries.put_nowait(frame)
            await self.push_frame(frame, direction)

    async def run_session(input_sink, *, on_transcript, **kwargs):
        delivered = asyncio.Event()

        async def accept(query):
            await input_sink(query)
            delivered.set()

        gate = VoiceGateProcessor(cfg=voice._session.voice_gate, tts=tts)
        voice_io = _VoiceIOProcessor(accept)
        voice._session._io_processor = voice_io
        pipeline = Pipeline([
            gate, voice_io,
            StreamingTtsProcessor(
                tts=tts, voice_gate=gate.gate, transport=transport, text_topic=voice._session.text_topic,
            ),
            OutputSink(),
        ])
        worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
        runner = WorkerRunner()
        await runner.add_workers(worker)
        started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def on_started(*args):
            started.set()

        async def utterance(text, acknowledgement=False):
            ts = time.time_ns() // 1000
            delivered.clear()
            if raw_transcript_first:
                previous_action = demo.capture._tails.get("user")
                await on_transcript("user", text, ts)
                await voice._transcript_queue.join()
                # Raw STT is for shared capture, never a second command path.
                assert demo.capture._tails.get("user") is previous_action
            frame = TranscriptionFrame(text=text, user_id="user", timestamp="")
            frame.transport_source = "user"
            frame.pts = ts * 1000
            await worker.queue_frame(frame)
            await delivered.wait()
            await asyncio.shield(demo.capture._tails["user"])
            if acknowledgement:
                boundary = await boundaries.get()
                assert boundary.transport_destination == "user"
            if not raw_transcript_first:
                previous_action = demo.capture._tails.get("user")
                await on_transcript("user", text, ts)
                await voice._transcript_queue.join()
                assert demo.capture._tails.get("user") is previous_action

        task = asyncio.create_task(runner.run())
        try:
            async with asyncio.timeout(5):
                await started.wait()
                await demo.capture.receive(joined())
                for _ in range(2):
                    await utterance("Start capture.", acknowledgement=True)
                    state = demo.capture_backend._sessions["user"]
                    await utterance("Place a cup")
                    await utterance("Stop capture.", acknowledgement=True)
                    await utterance("Stop capture.")  # Already stopped: no duplicate reply.
                    assert packet(state)["status"] == "complete"
                    assert narration(state) == ["Place a cup"]
                await demo.capture.close()
                await worker.queue_frame(EndFrame())
                await task
        finally:
            if not task.done():
                await worker.cancel()
            await asyncio.gather(task, return_exceptions=True)
            voice._session._io_processor = None

    monkeypatch.setattr(voice._session, "run", run_session)
    runtime = AgentRuntime()
    runtime.register("voice", voice)
    runtime.register("capture", demo.capture)
    async with runtime:
        await voice.run(runtime)

    expected = ["Capture started.", "Capture stopped."] * 2
    assert [call.args[0] for call in tts.synthesize.await_args_list] == expected
    assert audio_frames and all(frame.transport_destination == "user" for frame in audio_frames)
    echoes = [call.args[0].data.decode() for call in transport.send_return_data.await_args_list
              if call.args[0].topic == "agent.response"]
    assert echoes == expected
    transcripts = [call.args[0].data.decode() for call in transport.send_return_data.await_args_list
                   if call.args[0].topic == CAPTURE_STT_TOPIC]
    assert transcripts == ["Start capture.", "Place a cup", "Stop capture.", "Stop capture."] * 2


async def test_participants_are_independent(demo):
    for pid in ("user", "other"):
        await demo.capture.receive(joined(pid))
        await speak(demo, "start capture", pid)
        await speak(demo, pid + " narration", pid)
    other = demo.capture_backend._sessions["other"]
    await speak(demo, "stop capture")
    assert not demo.capture_backend.is_capturing("user")
    assert demo.capture_backend.is_capturing("other")
    assert narration(other) == ["other narration"]


@pytest.mark.parametrize("text", [
    "stop", "start recording", "stop recording", "do not stop capture", "say stop capture", "stop capture now",
    "start capture the next step", "do not start capture",
])
async def test_only_standalone_commands_control_recording(demo, text):
    await demo.capture.receive(joined())
    await speak(demo, text)
    assert not demo.capture_backend.is_capturing("user")
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await speak(demo, text)
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, "stop capture")
    assert narration(state) == [text]


async def test_failed_narration_write_cannot_seal_a_complete_packet(demo, monkeypatch):
    import task_worker.capture.capturing as module

    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    original = module._append_jsonl

    def fail_transcript(path, row):
        if path.name == "transcript.jsonl":
            raise OSError("write failed")
        original(path, row)

    monkeypatch.setattr(module, "_append_jsonl", fail_transcript)
    await speak(demo, "Place the cup")
    await speak(demo, "stop capture")
    assert packet(state)["status"] == "incomplete"
    assert packet(state)["narration_status"] == "failed"
    assert announcements(demo) == ["Capture started.", "Capture stopped."]


async def test_no_camera_frames_does_not_end_narration(demo):
    demo.frames.execute.side_effect = FrameUnavailable("no camera available")
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    async with asyncio.timeout(2):
        while not (state.directory / "errors.jsonl").exists():
            await asyncio.sleep(0.001)
    await speak(demo, "Explain the procedure without video")
    await speak(demo, "stop capture")
    assert packet(state)["status"] == "complete"
    assert packet(state)["counts"] == {"frames": 0, "transcripts": 1, "captions": 0}


@pytest.mark.parametrize("boundary", ["restart", "disconnect", "shutdown"])
async def test_queued_restart_and_narration_are_ordered_without_media_wait(demo, monkeypatch, boundary):
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    old = demo.capture_backend._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.capture_backend._finalize

    async def delayed(state, **kwargs):
        if state is old:
            entered.set()
            await release.wait()
        await original(state, **kwargs)

    monkeypatch.setattr(demo.capture_backend, "_finalize", delayed)
    await speak(demo, "First", wait=False)
    finishing = asyncio.create_task(speak(demo, "stop capture"))
    await asyncio.wait_for(entered.wait(), 2)
    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing
    await speak(demo, "start capture", wait=False)
    await speak(demo, "Second", wait=False)
    closing = None
    if boundary == "disconnect":
        closing = asyncio.ensure_future(demo.capture.receive(
            ParticipantEvent("user", False, time.time_ns() // 1000, participant_session_id="session")
        ))
    elif boundary == "shutdown":
        closing = asyncio.create_task(demo.capture.close())
        await asyncio.sleep(0)
    await demo.capture.receive(joined("other"))
    if boundary != "shutdown":
        await speak(demo, "start capture", "other")
    try:
        assert demo.capture_backend._sessions["user"] is old
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(demo.capture._tails["user"]), 2)
        if closing:
            await asyncio.wait_for(closing, 2)
    assert narration(old) == ["First"]
    if boundary == "restart":
        new = demo.capture_backend._sessions["user"]
        assert new is not old
        assert narration(new) == ["Second"]
    else:
        assert not demo.capture_backend.is_capturing("user")
        assert len(list((demo.root / "sessions").glob("*-user"))) == 1


async def test_on_demand_images_and_republished_stream_are_distinct(demo, monkeypatch):
    async def idle(_state):
        await asyncio.Future()

    monkeypatch.setattr(demo.capture_backend, "_capture_loop", idle)
    monkeypatch.setattr(demo.capture_backend, "_caption_loop", idle)
    await demo.capture.receive(joined())
    await speak(demo, 'start capture')
    state = demo.capture_backend._sessions["user"]
    image = demo.frames.execute.return_value
    # All on-demand snapshots use sequence zero. A live track can reset its
    # sequence too; neither should discard a different timestamp or source.
    for index, (track, timestamp, sequence) in enumerate([
        ("", 100, 0), ("", 200, 0), ("camera", 300, 0), ("replacement", 300, 0), ("", 400, 0),
    ], start=1):
        demo.frames.execute.return_value = image.model_copy(update={
            "track_id": track, "timestamp_us": timestamp, "sequence": sequence,
        })
        await demo.capture_backend._capture(state)
        await demo.capture_backend._caption(state)
        await demo.capture_backend._capture(state)  # Same observation is not new evidence.
        await demo.capture_backend._caption(state)
        assert state.frame_count == state.caption_count == index
    demo.frames.execute.side_effect = FrameUnavailable("live camera disabled")
    with pytest.raises(FrameUnavailable):
        await demo.capture_backend._capture(state)
    await demo.capture_backend._caption(state)
    assert state.caption_count == 5  # No repeated caption of the stale last frame.
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, 'stop capture')
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["counts"]["frames"] == packet["counts"]["captions"] == 5


@pytest.fixture
async def manual_caption_demo(demo, monkeypatch):
    async def idle(_state):
        await asyncio.Future()

    monkeypatch.setattr(demo.capture_backend, "_capture_loop", idle)
    monkeypatch.setattr(demo.capture_backend, "_caption_loop", idle)
    await demo.capture.receive(joined())
    return demo


@pytest.mark.parametrize("boundary", ["command", "disconnect", "shutdown"])
async def test_effective_end_preserves_frames_after_stop_utterance(manual_caption_demo, monkeypatch, boundary):
    from datetime import datetime

    import task_worker.capture.capturing as module

    demo = manual_caption_demo
    await speak(demo, "start capture", timestamp_us=1_000_000)
    state = demo.capture_backend._sessions["user"]
    assert packet(state)["ended_at"] is None
    # A frame arrives after the stop utterance was spoken, but before STT
    # delivers the recognized command. That frame remains legitimate evidence.
    demo.frames.execute.return_value = demo.frames.execute.return_value.model_copy(update={
        "timestamp_us": 3_000_000,
    })
    await demo.capture_backend._capture(state)
    clock = [4_000_000]
    monkeypatch.setattr(module, "_now_us", lambda: clock[0])
    drain = demo.capture_backend._drain_writes

    async def slow_drain(session):
        # The end describes capture admission, not when filesystem work ends.
        clock[0] = 9_000_000
        await drain(session)

    monkeypatch.setattr(demo.capture_backend, "_drain_writes", slow_drain)
    if boundary == "command":
        await speak(demo, "stop capture", timestamp_us=2_000_000)
    elif boundary == "disconnect":
        await demo.capture.receive(ParticipantEvent(
            "user", False, 4_000_000, participant_session_id="session",
        ))
    else:
        await demo.capture.close()
    saved = packet(state)
    assert saved["status"] == "complete"
    assert saved["counts"]["frames"] == 1
    assert datetime.fromisoformat(saved["ended_at"]).timestamp() == 4
    assert saved["stop_command"] == (
        {"text": "stop capture", "timestamp_us": 2_000_000} if boundary == "command" else None
    )
    records = [json.loads(line) for line in (state.directory / "frames/index.jsonl").read_text().splitlines()]
    assert records[0]["timestamp_us"] == 3_000_000


async def test_empty_packet_contract(manual_caption_demo):
    demo = manual_caption_demo
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await speak(demo, "stop capture")
    saved = packet(state)
    assert saved["schema_version"] == 1
    assert saved["status"] == saved["narration_status"] == "complete"
    assert saved["counts"] == {"frames": 0, "transcripts": 0, "captions": 0}
    assert (state.directory / "summary.md").is_file()
    assert (state.directory / "transcript.jsonl").read_text() == ""
    assert not (state.directory / "frames/index.jsonl").exists()
    assert not (state.directory / "captions.jsonl").exists()
    assert not (state.directory / "errors.jsonl").exists()
    assert all(not Path(path).is_absolute() for path in saved["files"].values())


async def test_caption_uses_saved_image_pairs_in_order_and_resets(manual_caption_demo):
    from xr_ai_tools.vision import MultiImageQueryTool

    demo = manual_caption_demo
    prompt = load_config(_SAMPLE / "yaml/worker.yaml").capture.caption_prompt
    response = demo.captioner.execute.return_value.text
    vlm = Mock(ask_images=AsyncMock(return_value=SimpleNamespace(content=response)))
    demo.capture_backend._query_images = MultiImageQueryTool(
        images=demo.capture_backend._images, vlm=vlm, system_prompt=prompt,
    )
    source = demo.frames.execute.return_value
    for _ in range(2):
        await speak(demo, "start capture")
        state = demo.capture_backend._sessions["user"]
        paths = []
        for timestamp in (100, 200, 300):
            demo.frames.execute.return_value = source.model_copy(update={"timestamp_us": timestamp})
            await demo.capture_backend._capture(state)
            paths.append(state.directory / state.latest_frame.path)
            if timestamp == 200:
                continue  # Sampling is faster than captioning; compare captioned frames.
            await demo.capture_backend._caption(state)
        initial_call = vlm.ask_images.await_args_list[-2]
        assert initial_call.args[0] == [paths[0]]
        assert "exactly ONE image" in initial_call.args[1]
        assert "set delta to Initial observation" in initial_call.args[1]
        assert "Previous grouping labels" not in initial_call.args[1]
        call = vlm.ask_images.await_args_list[-1]
        assert call.args[0] == [paths[0], paths[2]]
        assert all(path.is_file() for path in call.args[0])
        assert call.kwargs["system_prompt"] == prompt
        assert "A blue block" not in call.args[1]  # No previous caption text.
        assert "earlier" in call.args[1] and "current" in call.args[1]
        assert "exactly TWO images" in call.args[1]
        assert "FIRST=earlier, SECOND=current" in call.args[1]
        assert "Earlier: ... Current: ... Change: ..." in call.args[1]
        assert "Do not use Initial observation" in call.args[1]
        # Re-resolve from saved snapshots even after the live registry evicts them.
        demo.frames.execute.return_value = source.model_copy(update={"timestamp_us": 400})
        await demo.capture_backend._capture(state)
        current = state.directory / state.latest_frame.path
        demo.capture_backend._images.clear()
        await demo.capture_backend._caption(state)
        assert vlm.ask_images.await_args.args[0] == [paths[2], current]
        await demo.capture_backend._caption(state)  # No additional call for a stale image.
        assert state.caption_count == 3
        await speak(demo, "stop capture")
        assert packet(state)["counts"] == {"frames": 4, "captions": 3, "transcripts": 0}
        captions = [json.loads(line) for line in (state.directory / "captions.jsonl").read_text().splitlines()]
        assert [(c["frame_id"], c["delta_reference_frame_id"], c["delta_reference_timestamp_us"])
                for c in captions] == [(1, None, None), (3, 1, 100), (4, 3, 300)]
        # The next recording gets fresh live references, not the previous session's pair.
        source = source.model_copy(update={"image": demo.capture_backend._images.put(paths[0].read_bytes())})
    assert vlm.ask_images.await_count == 6


@pytest.mark.parametrize("failure", ["unavailable", "invalid_json"])
async def test_failed_caption_does_not_advance_pair_anchor(manual_caption_demo, failure):
    demo = manual_caption_demo
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    source = demo.frames.execute.return_value
    success = demo.captioner.execute.return_value
    paths = []
    for timestamp in (100, 200, 300):
        demo.frames.execute.return_value = source.model_copy(update={"timestamp_us": timestamp})
        await demo.capture_backend._capture(state)
        paths.append(state.directory / state.latest_frame.path)
        if timestamp == 200:
            demo.captioner.execute.return_value = SimpleNamespace(
                available=failure != "unavailable", text="VLM unavailable" if failure == "unavailable" else "{",
            )
            with pytest.raises(RuntimeError if failure == "unavailable" else ValueError):
                await demo.capture_backend._caption(state)
            assert state.last_captioned_frame.frame_id == 1
            demo.captioner.execute.return_value = success
        else:
            await demo.capture_backend._caption(state)
    request = demo.captioner.execute.await_args.args[0]
    assert [demo.capture_backend._images.resolve(image) for image in request.images] == [paths[0], paths[2]]
    assert state.caption_count == 2
    assert state.last_captioned_frame.frame_id == 3
    captions = [json.loads(line) for line in (state.directory / "captions.jsonl").read_text().splitlines()]
    assert captions[-1]["delta_reference_frame_id"] == 1
    assert captions[-1]["delta_reference_timestamp_us"] == 100


async def test_stop_during_caption_append_seals_consistent_hierarchy(demo, monkeypatch):
    import task_worker.capture.capturing as module

    entered, release = threading.Event(), threading.Event()
    original = module._append_jsonl

    def delayed(path, record):
        if path.name == "captions.jsonl":
            entered.set()
            assert release.wait(5)
        original(path, record)

    monkeypatch.setattr(module, "_append_jsonl", delayed)
    await demo.capture.receive(joined())
    await speak(demo, 'start capture')
    state = demo.capture_backend._sessions["user"]
    finishing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        finishing = asyncio.create_task(speak(demo, 'stop capture'))
        async with asyncio.timeout(2):
            while not any(task.cancelled() for task in state.tasks):
                await asyncio.sleep(0.001)
        assert not finishing.done()
    finally:
        release.set()
        if finishing is not None:
            await asyncio.wait_for(finishing, 2)

    captions = [json.loads(line) for line in (state.directory / "captions.jsonl").read_text().splitlines()]
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["status"] == "complete"
    assert packet["counts"]["captions"] == len(captions) == 1
    assert len(packet["hierarchy"]) == 1
    activity = packet["hierarchy"][0]
    assert activity["caption_count"] == 1
    assert activity["phases"][0]["summary"] == captions[0]["caption"]
    summary = (state.directory / "summary.md").read_text()
    assert "Waiting for the first visual caption" not in summary
    assert captions[0]["caption"] in summary


async def test_shutdown_drains_inflight_writes_before_final_packet(demo, monkeypatch):
    import task_worker.capture.capturing as module

    await demo.capture.receive(joined())
    await speak(demo, 'start capture')
    state = demo.capture_backend._sessions["user"]
    entered, release = threading.Event(), threading.Event()
    original = module._atomic_json

    def write(path, packet):
        if packet["status"] == "recording":
            entered.set()
            assert release.wait(5)
        original(path, packet)

    monkeypatch.setattr(module, "_atomic_json", write)

    async def write_view():
        async with state.lock:
            await demo.capture_backend._write_packet(state)

    state.tasks.append(asyncio.create_task(write_view()))
    assert await asyncio.to_thread(entered.wait, 2)
    closing = asyncio.create_task(demo.capture.close())
    try:
        await asyncio.sleep(0.02)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert json.loads((state.directory / "packet.json").read_text())["status"] == "complete"
    assert not demo.capture_backend.is_capturing("user")


@pytest.mark.parametrize("response", [
    "", "A cup is on the table.", "{", "[]",
    '<think>Reasoning with {"activity":"Setup","phase":"Arrange","caption":"A cup","delta":"Initial"}',
    '{"caption":"A cup"}',
    '{"activity":null,"phase":"Arrange","caption":"A cup","delta":"Initial"}',
    '{"activity":"Setup","phase":[],"caption":"A cup","delta":"Initial"}',
    '{"activity":"Setup","phase":"Arrange","caption":" ","delta":"Initial"}',
])
async def test_invalid_caption_is_error_not_evidence(demo, response):
    demo.captioner.execute.return_value = SimpleNamespace(available=True, text=response)
    demo.capture_backend._caption_interval_s = 0.01
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    async with asyncio.timeout(2):
        while not (state.directory / "errors.jsonl").exists():
            await asyncio.sleep(0.001)
    assert state.last_captioned_frame is None
    assert state.previous_caption is None
    await speak(demo, "stop capture")
    assert packet(state)["counts"]["captions"] == 0
    assert packet(state)["hierarchy"] == []
    assert not (state.directory / "captions.jsonl").exists()
    errors = [json.loads(line) for line in (state.directory / "errors.jsonl").read_text().splitlines()]
    assert any(row["kind"] == "caption_error" and "raw response:" in row["message"] for row in errors)


def testcaption_query_treats_previous_labels_as_hints_not_visual_evidence():
    previous = {
        "activity": "Observing", "phase": "Initial observation",
        "caption": "Unverified attachment", "delta": "Unverified completion",
    }
    query = caption_query(previous)
    assert "exactly TWO images" in query
    assert "Do not use Initial observation" in query
    prefix, labels = query.split("\nPrevious grouping labels ", 1)
    assert "broad parent activity" in prefix and "specific current phase" in prefix
    assert labels.startswith("(fallible naming hints, not visual evidence): ")
    assert json.loads(labels.split(": ", 1)[1]) == {
        "activity": "Observing", "phase": "Initial observation",
    }
    assert "Unverified" not in query


async def test_caption_grouping_uses_labels_only_and_preserves_transitions(manual_caption_demo):
    demo = manual_caption_demo
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    source = demo.frames.execute.return_value
    groups = [("Build", "Align"), ("Build", "Align"), ("Build", "Fasten"), ("Clean", "Wipe")]
    for index, (activity, phase) in enumerate(groups, 1):
        demo.frames.execute.return_value = source.model_copy(update={"timestamp_us": index})
        await demo.capture_backend._capture(state)
        demo.captioner.execute.return_value.text = json.dumps({
            "activity": activity, "phase": phase, "caption": f"Observation {index}", "delta": f"Change {index}",
        })
        await demo.capture_backend._caption(state)
        query = demo.captioner.execute.await_args.args[0].query
        if index > 1:
            assert json.dumps({"activity": groups[index-2][0], "phase": groups[index-2][1]}) in query
        else:
            assert "Previous grouping labels" not in query
        assert f"Observation {index - 1}" not in query
        assert f"Change {index - 1}" not in query
    await speak(demo, "stop capture")
    activities = packet(state)["hierarchy"]
    assert [a["name"] for a in activities] == ["Build", "Clean"]
    assert [a["caption_count"] for a in activities] == [3, 1]
    assert [p["name"] for p in activities[0]["phases"]] == ["Align", "Fasten"]
    assert activities[0]["phases"][0]["caption_count"] == 2


@pytest.mark.parametrize("stage", ["start", "stop"])
@pytest.mark.parametrize("target", ["transcript.jsonl", "packet.json", "summary.md"])
async def test_storage_failure_releases_packet_and_allows_fresh_start(manual_caption_demo, monkeypatch, stage, target):
    import task_worker.capture.capturing as module

    demo = manual_caption_demo
    if stage == "stop":
        await speak(demo, "start capture")
        await speak(demo, "First procedure")
    original = module._atomic_text

    def fail(path, text):
        if path.name == target:
            raise OSError("disk full")
        original(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_atomic_text", fail)
        if stage == "start":
            await speak(demo, "start capture")
        else:
            # A missing narration log exercises the final read failure as well.
            if target == "transcript.jsonl":
                state = demo.capture_backend._sessions["user"]
                (state.directory / "transcript.jsonl").unlink()
            await speak(demo, "stop capture")
    assert not demo.capture_backend.is_capturing("user")
    assert "user" not in demo.capture_backend._sessions
    assert "user" not in demo.capture._sessions
    assert demo.capture._connections["user"].recording is None
    old = next((demo.root / "sessions").iterdir())
    if (old / "packet.json").exists():
        assert json.loads((old / "packet.json").read_text())["status"] != "complete"
    await speak(demo, "start capture")
    new = demo.capture_backend._sessions["user"]
    assert new.directory != old
    await speak(demo, "Second procedure")
    await speak(demo, "stop capture")
    assert narration(new) == ["Second procedure"]
    assert packet(new)["status"] == "complete"


@pytest.mark.parametrize("failing_file", ["index.jsonl", "captions.jsonl", "errors.jsonl"])
async def test_background_write_failures_are_incomplete_and_loops_survive(demo, monkeypatch, failing_file):
    import task_worker.capture.capturing as module

    original = module._append_jsonl
    attempted = threading.Event()

    def fail(path, row):
        if path.name == failing_file:
            attempted.set()
            raise OSError("disk full")
        original(path, row)

    monkeypatch.setattr(module, "_append_jsonl", fail)
    if failing_file == "errors.jsonl":
        demo.frames.execute.side_effect = FrameUnavailable("camera unavailable")
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    assert await asyncio.to_thread(attempted.wait, 2)
    await asyncio.sleep(0.05)
    assert all(not task.done() for task in state.tasks)
    await speak(demo, "stop capture")
    assert packet(state)["status"] == "incomplete"


async def test_stop_preserves_live_frame_cache_until_departure(demo):
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    await speak(demo, "stop capture")
    demo.frames.release.assert_not_called()
    await speak(demo, "start capture")
    await demo.capture.receive(ParticipantEvent("user", False, 2, participant_session_id="session"))
    demo.frames.release.assert_called_once_with("user")


@pytest.mark.parametrize("boundary", ["disconnect", "shutdown"])
async def test_departure_during_start_directory_write_drains_without_narration_error(demo, monkeypatch, boundary):
    original = Path.mkdir
    entered, release = threading.Event(), threading.Event()

    def delayed(path, *args, **kwargs):
        if path.name == "frames":
            entered.set()
            assert release.wait(5)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", delayed)
    await demo.capture.receive(joined())
    await speak(demo, "start capture", wait=False)
    assert await asyncio.to_thread(entered.wait, 2)
    state = demo.capture_backend._sessions["user"]
    closing = asyncio.ensure_future(
        demo.capture.close() if boundary == "shutdown" else demo.capture.receive(
            ParticipantEvent("user", False, 2, participant_session_id="session"),
        ),
    )
    try:
        await asyncio.sleep(0.01)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert not demo.capture_backend._sessions
    assert not demo.capture._sessions
    assert announcements(demo) == []
    assert packet(state)["status"] == "incomplete"
    assert packet(state)["counts"] == {"frames": 0, "captions": 0, "transcripts": 0}
    assert not (state.directory / "errors.jsonl").exists()


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "```\n{}\n```"])
def test_caption_parser_accepts_only_complete_structured_observations(wrapper):
    from task_worker.capture.captioning import parse_caption

    value = {"activity": "Setup", "phase": "Align", "caption": "A cup", "delta": "Initial observation"}
    assert parse_caption(wrapper.format(json.dumps(value))) == value


async def test_failed_pending_caption_write_cannot_seal_complete(demo, monkeypatch):
    import task_worker.capture.capturing as module

    original = module._append_jsonl
    entered, release = threading.Event(), threading.Event()

    def delayed_failure(path, row):
        if path.name == "captions.jsonl":
            entered.set()
            assert release.wait(5)
            raise OSError("late filesystem failure")
        original(path, row)

    monkeypatch.setattr(module, "_append_jsonl", delayed_failure)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    assert await asyncio.to_thread(entered.wait, 2)
    closing = asyncio.create_task(speak(demo, "stop capture"))
    try:
        async with asyncio.timeout(2):
            while not any(task.cancelled() for task in state.tasks):
                await asyncio.sleep(0.001)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert packet(state)["status"] == "incomplete"
    assert packet(state)["counts"]["captions"] == 0
    assert not demo.capture_backend._sessions


async def test_failed_stop_does_not_clear_queued_restart(manual_caption_demo, monkeypatch):
    import task_worker.capture.capturing as module

    demo = manual_caption_demo
    await speak(demo, "start capture")
    old = demo.capture_backend._sessions["user"]
    entered, release = threading.Event(), threading.Event()
    original = module._atomic_text

    def fail_old_summary(path, text):
        if path.parent == old.directory and path.name == "summary.md":
            entered.set()
            assert release.wait(5)
            raise OSError("cannot seal old summary")
        original(path, text)

    monkeypatch.setattr(module, "_atomic_text", fail_old_summary)
    await speak(demo, "stop capture", wait=False)
    assert await asyncio.to_thread(entered.wait, 2)
    try:
        await speak(demo, "start capture", wait=False)
        await speak(demo, "New narration", wait=False)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(demo.capture._tails["user"]), 2)
    new = demo.capture_backend._sessions["user"]
    assert new is not old
    assert demo.capture._connections["user"].recording is demo.capture._sessions["user"]
    assert narration(new) == ["New narration"]
    assert narration(old) == []
    assert packet(old)["status"] == "incomplete"
