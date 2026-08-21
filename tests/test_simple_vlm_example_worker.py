# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the packaged simple-vlm-example worker."""

from __future__ import annotations

import asyncio
import io
import sys
import time
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import nemo_relay
import pytest
import yaml
from PIL import Image
from xr_ai_hub import FrameData, FrameSignal, FrameUnavailable, PixelFormat, ProcessorEndpoint
from xr_ai_models import ChatResponse, VLMService
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import UserQuery, VoiceAgent, VoiceInterrupted, VoiceOutput
from xr_ai_voice import _runtime as voice_runtime_module
from xr_ai_voice._readiness import wait_for_services
from xr_ai_voice._session import _VoiceSession as VoiceSession
from xr_ai_voicegate import VoiceGateConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "simple-vlm-example"
_WORKER_DIR = _SAMPLE_DIR / "worker"
sys.path.insert(0, str(_WORKER_DIR))

from simple_vlm_example_worker import __main__ as worker_main  # noqa: E402  # pyright: ignore[reportMissingImports]
from simple_vlm_example_worker import app  # noqa: E402  # pyright: ignore[reportMissingImports]
from simple_vlm_example_worker.agent import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    INTERRUPTED_TOPIC,
    USER_QUERY_TOPIC,
    SimpleVlmAgent,
)
from simple_vlm_example_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from xr_ai_tools import Tool  # noqa: E402
from xr_ai_tools import current_frame as current_frame_module  # noqa: E402
from xr_ai_tools.current_frame import (  # noqa: E402
    CurrentFrameRequest,
    CurrentFrameTool,
    ImageFrame,
)
from xr_ai_tools.image import ImageReference, ImageRegistry  # noqa: E402
from xr_ai_tools.vision import (  # noqa: E402
    ImageQueryRequest,
    ImageQueryResult,
    ImageQueryTool,
    StreamingImageQueryTool,
)


class _Service:
    def __init__(self) -> None:
        self.health_calls = 0
        self.close_calls = 0
        self.stream_calls = []

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def stream_images(
        self,
        images,
        question,
        *,
        system_prompt="",
        max_tokens=None,
        timeout=None,
    ):
        self.stream_calls.append(
            (images, question, system_prompt, max_tokens, timeout)
        )
        yield "gray"

    async def close(self) -> None:
        self.close_calls += 1


class _FlakyWarmupService(_Service):
    async def stream_images(
        self,
        images,
        question,
        *,
        system_prompt="",
        max_tokens=None,
        timeout=None,
    ):
        self.stream_calls.append((images, question, system_prompt, max_tokens, timeout))
        if len(self.stream_calls) == 1:
            raise RuntimeError("still loading")
        yield "gray"


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _DataEndpoint()
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _DataEndpoint:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str]] = []

    def on_audio(self, callback):
        self.audio_callback = callback

        def unsubscribe() -> None:
            self.audio_callback = None

        return unsubscribe

    def on_data(self, callback) -> None:
        self.data_callback = callback

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))


class _CurrentFrameTool:
    instances: list["_CurrentFrameTool"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.released: list[str] = []
        self.instances.append(self)

    async def execute(self, request):
        self.requests.append(request)
        return SimpleNamespace(image=ImageReference(uri="https://example.com/frame.jpg"))

    def release(self, participant_id: str) -> None:
        self.released.append(participant_id)


class _StreamingImageQueryTool:
    instances: list["_StreamingImageQueryTool"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.instances.append(self)

    async def stream(self, request):
        self.requests.append(request)
        for text in ("a ", "blue square"):
            yield SimpleNamespace(text=text)


class _SelectedFrameTool:
    async def execute(self, _request):
        return SimpleNamespace(image=ImageReference(uri="https://example.com/frame.jpg"))

    def release(self, _participant_id: str) -> None:
        pass


async def _ignore_status(_status: str, _participant_id: str) -> None:
    pass


class _LiveEndpoint:
    def __init__(self) -> None:
        self.frame_callback = None
        self.statuses: list[tuple[str, str]] = []

    def on_frame(self, callback) -> None:
        self.frame_callback = callback

    def on_participant(self, _callback) -> None:
        return None

    async def request_frame(self, signal: FrameSignal) -> FrameData:
        return FrameData(
            seq=signal.seq,
            pts_us=signal.pts_us,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data=bytes([20, 40, 60] * 4),
            participant_id=signal.participant_id,
            track_id=signal.track_id,
        )

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))


class _StreamingVlm:
    def __init__(self) -> None:
        self.calls = []
        self.ask_calls = []

    async def ask_images(
        self,
        images,
        question: str,
        *,
        system_prompt: str = "",
        headers=None,
    ) -> ChatResponse:
        self.ask_calls.append((images, question, system_prompt, dict(headers or {})))
        return ChatResponse("a blue square", None, None, "stop", {})

    async def stream_images(
        self,
        images,
        question: str,
        *,
        system_prompt: str = "",
        headers=None,
    ):
        self.calls.append((images, question, system_prompt, dict(headers or {})))
        for token in ("a ", "blue ", "square"):
            yield token


def test_worker_is_a_package_with_module_and_console_entry_points() -> None:
    project = tomllib.loads((_WORKER_DIR / "pyproject.toml").read_text())
    package = _WORKER_DIR / "simple_vlm_example_worker"
    dependencies = set(project["project"]["dependencies"])

    assert project["project"]["scripts"]["simple_vlm_example_worker"] == ("simple_vlm_example_worker.__main__:run")
    assert "xr-ai-hub-client" in dependencies
    assert "xr-ai-agent-runtime" in dependencies
    assert project["tool"]["uv"]["sources"]["xr-ai-agent-runtime"]["path"] == (
        "../../../agent-sdk/xr-ai-runtime"
    )
    assert (
        _WORKER_DIR
        / project["tool"]["uv"]["sources"]["xr-ai-agent-runtime"]["path"]
    ).resolve().is_dir()
    assert project["tool"]["uv"]["sources"]["xr-ai-hub-client"]["path"] == (
        "../../../agent-sdk/xr-ai-hub"
    )
    assert (
        _WORKER_DIR / project["tool"]["uv"]["sources"]["xr-ai-hub-client"]["path"]
    ).resolve().is_dir()
    assert "xr-ai-tools[frames,vision]" in dependencies
    assert all("[vision" not in dependency and "[voice" not in dependency for dependency in dependencies)
    assert "xr-ai-voice" in dependencies
    assert "xr-ai-pipecat" not in dependencies
    assert all("mcp" not in dependency.lower() for dependency in dependencies)
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["simple_vlm_example_worker"]
    assert {
        "__init__.py",
        "__main__.py",
        "agent.py",
        "app.py",
        "config.py",
        "prompts/system.txt",
    } <= {str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()}
    assert not (_WORKER_DIR / "agent.py").exists()
    assert not (_WORKER_DIR / "simple_vlm_example_worker.py").exists()


def test_entry_point_loads_config_and_forwards_ready_file(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "worker.yaml"
    ready_file = tmp_path / "ready"
    config = object()
    seen = {}

    def fake_load_config(path):
        seen["config_path"] = path
        return config

    monkeypatch.setattr(worker_main, "load_config", fake_load_config)

    async def fake_run_app(loaded_config, *, ready_file):
        seen["config"] = loaded_config
        seen["ready_file"] = ready_file

    monkeypatch.setattr(worker_main, "run_app", fake_run_app)
    worker_main.run(
        [
            "--config",
            str(config_path),
            "--ready-file",
            str(ready_file),
            "--launcher-option",
            "ignored",
        ]
    )

    assert seen == {
        "config_path": config_path,
        "config": config,
        "ready_file": ready_file,
    }


def test_shipped_config_preserves_models_and_prompt_behavior() -> None:
    config_path = _SAMPLE_DIR / "yaml" / "simple_vlm_example_worker.yaml"
    config = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text())
    prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "system.txt").read_text()

    assert config.models_config == _SAMPLE_DIR / "yaml" / "models.json"
    assert config.voice_gate_yaml == _SAMPLE_DIR / "yaml" / "voice_gate.yaml"
    assert config.system_prompt == prompt
    assert "Speak directly to me in second person" in prompt
    assert 'Never refer to "the user" in the third person.' in prompt
    assert "when the user" not in prompt
    assert "If the user" not in prompt
    assert config.frame_max_age_s == 5.0
    assert config.frame_timeout_s == 5.0
    assert config.idle_timeout_secs is None
    assert "system_prompt_file" not in raw


def test_config_without_a_file_uses_packaged_defaults(tmp_path) -> None:
    prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "system.txt").read_text()

    for config_path in (None, tmp_path / "missing.yaml"):
        config = load_config(config_path)
        assert config.system_prompt == prompt
        expected_parent = Path() if config_path is None else tmp_path
        assert config.models_config == expected_parent / "models.json"


def test_blank_inline_prompt_uses_packaged_default(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("system_prompt:\n")
    prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "system.txt").read_text()

    assert load_config(config_path).system_prompt == prompt


def test_config_keeps_inline_prompt_compatibility(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "voice_gate_yaml: gate.yaml\n"
        "system_prompt: custom prompt\n"
        "idle_timeout_secs: 30\n"
    )

    config = load_config(config_path)

    assert config.models_config == tmp_path / "models.json"
    assert config.voice_gate_yaml == tmp_path / "gate.yaml"
    assert config.system_prompt == "custom prompt"
    assert config.idle_timeout_secs == 30.0


def test_config_rejects_a_non_mapping_yaml_document(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(config_path)


async def test_simple_vlm_agent_closes_tool_stream_when_publication_fails() -> None:
    closed = asyncio.Event()

    class Stream:
        def __init__(self) -> None:
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                await asyncio.Event().wait()
            self.sent = True
            return SimpleNamespace(text="first")

        async def aclose(self) -> None:
            closed.set()

    class Vision:
        def stream(self, _request):
            return Stream()

    class Context:
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, *_args, **_kwargs) -> None:
            raise RuntimeError("runtime stopped")

    agent = SimpleVlmAgent(
        lambda: (_SelectedFrameTool(), Vision()),  # type: ignore[return-value]
        _ignore_status,
    )
    with pytest.raises(RuntimeError, match="runtime stopped"):
        await agent._stream(  # noqa: SLF001
            UserQuery(text="What is shown?", timestamp_us=123),
            Context(),  # type: ignore[arg-type]
        )

    assert closed.is_set()


async def test_simple_vlm_agent_reports_relay_wrapped_missing_camera_frame() -> None:
    published: list[VoiceOutput] = []

    async def missing_frame(_request: CurrentFrameRequest) -> ImageFrame:
        raise FrameUnavailable("No camera frame available — please try again.")

    missing_frame_tool = Tool(
        "missing_camera_frame",
        "Reproduce a missing camera frame through the real Relay boundary.",
        CurrentFrameRequest,
        ImageFrame,
        missing_frame,
    )
    with pytest.raises(RuntimeError, match="internal error: FrameUnavailable") as wrapped:
        await missing_frame_tool.execute(CurrentFrameRequest(participant_id="alice"))
    assert wrapped.value.__cause__ is None
    assert wrapped.value.__context__ is None

    class Context:
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    agent = SimpleVlmAgent(
        lambda: (missing_frame_tool, _StreamingImageQueryTool()),  # type: ignore[return-value]
        _ignore_status,
    )

    await agent._stream(  # noqa: SLF001
        UserQuery(text="What is shown?", timestamp_us=123),
        Context(),  # type: ignore[arg-type]
    )

    assert [output.text for output in published] == [
        "No camera frame available — please try again.",
        "",
    ]
    assert [output.final for output in published] == [False, True]


async def test_cancelled_vlm_turn_does_not_publish_stream_terminator() -> None:
    waiting = asyncio.Event()
    closed = asyncio.Event()
    published: list[VoiceOutput] = []

    class Stream:
        def __init__(self) -> None:
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent:
                self.sent = True
                return SimpleNamespace(text="first")
            waiting.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closed.set()

    class Vision:
        def stream(self, _request):
            return Stream()

    class Context:
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    factory_calls: list[None] = []
    agent = SimpleVlmAgent(
        lambda: factory_calls.append(None) or (_SelectedFrameTool(), Vision()),  # type: ignore[return-value]
        _ignore_status,
    )
    assert factory_calls == []

    task = asyncio.create_task(
        agent._stream(  # noqa: SLF001
            UserQuery(text="What is shown?", timestamp_us=123),
            Context(),  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(waiting.wait(), 1.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert factory_calls == [None]
    assert closed.is_set()
    assert len(published) == 1
    assert published[0].final is False


async def test_vlm_warmup_failure_is_retried_by_readiness() -> None:
    vlm = _FlakyWarmupService()

    await asyncio.wait_for(
        wait_for_services(
            {"vlm": lambda: app._warm_vlm(cast(VLMService, vlm))},
            poll_interval=0,
        ),
        timeout=1,
    )

    assert vlm.health_calls == 2
    assert len(vlm.stream_calls) == 2


async def test_app_wires_text_voice_cleanup_readiness_and_shutdown(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_config(_SAMPLE_DIR / "yaml" / "simple_vlm_example_worker.yaml")
    ready_file = tmp_path / "ready"
    stt = _Service()
    vlm = _Service()
    tts = _Service()
    transport = _Transport()
    sessions: list[VoiceSession] = []
    run_options = {}
    responses = []
    response_tasks: list[asyncio.Task[None]] = []
    response_complete = asyncio.Event()

    worker_log = tmp_path / "logs" / "worker.log"
    worker_log.parent.mkdir()
    monkeypatch.setattr(app, "setup_logging", lambda _name: worker_log)
    monkeypatch.setattr(app, "load_models_config", lambda path: path)
    monkeypatch.setattr(app, "load_voice_gate_config", lambda _path: VoiceGateConfig())
    monkeypatch.setattr(app, "make_stt", lambda _models, _name: stt)
    monkeypatch.setattr(app, "make_vlm", lambda _models, _name: vlm)
    monkeypatch.setattr(app, "make_tts", lambda _models, _name: tts)
    monkeypatch.setattr(app, "HubVoiceTransport", lambda: transport)
    monkeypatch.setattr(app, "CurrentFrameTool", _CurrentFrameTool)
    monkeypatch.setattr(app, "StreamingImageQueryTool", _StreamingImageQueryTool)

    def make_session(**kwargs):
        session = VoiceSession(**kwargs)  # type: ignore[arg-type]
        sessions.append(session)

        async def run(handler, **options) -> None:
            if session.ready_file:
                session.ready_file.touch()
            run_options.update(options)
            assert (
                await handler(
                    SimpleNamespace(
                        participant_id="alice",
                        text="What is in front of me?",
                        timestamp_us=123,
                        interrupted_output=False,
                    )
                )
                is None
            )
            await asyncio.wait_for(response_complete.wait(), 1.0)
            await options["on_participant_left"]("alice")

            async def wait_until_released() -> None:
                while not _CurrentFrameTool.instances[0].released:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_released(), 1.0)

        async def enqueue_response(
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            interrupt: bool = False,
            pts_us: int | None = None,
        ) -> None:
            async def consume() -> None:
                text = response if isinstance(response, str) else "".join([chunk async for chunk in response])
                responses.append((participant_id, text, interrupt, pts_us))
                response_complete.set()

            response_tasks.append(asyncio.create_task(consume()))

        session.run = run  # type: ignore[method-assign]
        session.enqueue_response = enqueue_response  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(voice_runtime_module, "_VoiceSession", make_session)
    _CurrentFrameTool.instances.clear()
    _StreamingImageQueryTool.instances.clear()

    await app.run_app(config, ready_file=ready_file)

    assert ready_file.exists()
    relay_log = worker_log.parent / "relay-events.jsonl"
    assert relay_log.exists()
    relay_events = [yaml.safe_load(line) for line in relay_log.read_text().splitlines()]
    assert {event["name"] for event in relay_events} >= {
        "publish:simple-vlm.user-query",
        "agent:simple-vlm",
        "simple-vlm.turn",
        "voice.response",
    }
    assert "publish:voice.output" not in {event["name"] for event in relay_events}
    assert "agent:voice" not in {event["name"] for event in relay_events}
    assert stt.health_calls == tts.health_calls == vlm.health_calls == 1
    assert len(vlm.stream_calls) == 1
    warmup_images, question, system_prompt, max_tokens, timeout = vlm.stream_calls[0]
    assert len(warmup_images) == 1
    with Image.open(io.BytesIO(warmup_images[0])) as warmup_image:
        assert warmup_image.format == "JPEG"
        assert warmup_image.size == (1280, 720)
    assert question == "What is the dominant color?"
    assert system_prompt == "Answer with one word."
    assert max_tokens == 4
    assert timeout == 120.0
    assert stt.close_calls == tts.close_calls == vlm.close_calls == 1
    assert transport.shutdown_calls == 1
    assert sessions[0].text_topic == "vlm.response"
    assert _CurrentFrameTool.instances[0].kwargs["endpoint"] is transport.endpoint
    assert _CurrentFrameTool.instances[0].kwargs["frame_max_age_s"] == (config.frame_max_age_s)
    assert _CurrentFrameTool.instances[0].kwargs["frame_timeout_s"] == (config.frame_timeout_s)
    assert _CurrentFrameTool.instances[0].released == ["alice"]
    assert _CurrentFrameTool.instances[0].requests[0].participant_id == "alice"
    assert _StreamingImageQueryTool.instances[0].kwargs["vlm"] is vlm
    assert _StreamingImageQueryTool.instances[0].kwargs["system_prompt"] == (config.system_prompt)
    assert _StreamingImageQueryTool.instances[0].requests[0].query == ("What is in front of me?")
    assert _StreamingImageQueryTool.instances[0].requests[0].image.uri == ("https://example.com/frame.jpg")
    assert responses == [("alice", "a blue square", True, 123)]
    assert all(task.done() for task in response_tasks)
    assert run_options["interrupt_on_supersede"] is True
    assert callable(run_options["on_interrupted"])


async def test_relay_event_log_excludes_stream_chunks(tmp_path) -> None:
    worker_log = tmp_path / "worker.log"

    async with app._relay_event_log(worker_log):  # noqa: SLF001
        with nemo_relay.scope.scope("test-turn", nemo_relay.ScopeType.Agent):
            nemo_relay.scope.event("llm.chunk", data={"text": "fragment"})
            nemo_relay.scope.event("turn.summary", data={"text": "complete"})

    events = [yaml.safe_load(line) for line in (tmp_path / "relay-events.jsonl").read_text().splitlines()]
    names = [event["name"] for event in events]
    assert "llm.chunk" not in names
    assert "turn.summary" in names
    assert names.count("test-turn") == 2


async def test_selected_frame_returns_a_complete_agent_observation() -> None:
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=images,
    )
    vision = ImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )
    assert endpoint.frame_callback is not None
    await endpoint.frame_callback(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=time.time_ns() // 1_000,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data_sz=12,
            participant_id="alice",
            track_id="camera",
        )
    )
    events = []
    subscriber = "simple-vlm-finite-vision"
    intercept = "simple-vlm-finite-vision-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-8"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        frame = await frames.execute(CurrentFrameRequest(participant_id="alice"))
        result = await vision.execute(
            ImageQueryRequest(image=frame.image, query="What is shown?"),
        )
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert result == ImageQueryResult(text="a blue square")
    images, question, system_prompt, headers = vlm.ask_calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-8"
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_selected_frame_streams_typed_image_query_chunks() -> None:
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=images,
    )
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )
    assert endpoint.frame_callback is not None
    await endpoint.frame_callback(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=time.time_ns() // 1_000,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data_sz=12,
            participant_id="alice",
            track_id="camera",
        )
    )
    events = []
    subscriber = "simple-vlm-image-query"
    intercept = "simple-vlm-image-query-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-7"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        frame = await frames.execute(CurrentFrameRequest(participant_id="alice"))
        chunks = [chunk async for chunk in vision.stream(ImageQueryRequest(image=frame.image, query="What is shown?"))]
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert [chunk.text for chunk in chunks] == ["a ", "blue ", "square"]
    images, question, system_prompt, headers = vlm.calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-7"
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_streaming_image_query_stops_after_a_partial_failure() -> None:
    class _PartialFailureVlm:
        def __init__(self) -> None:
            self.calls = []

        async def stream_images(
            self,
            images,
            question: str,
            *,
            system_prompt: str = "",
            headers=None,
        ):
            self.calls.append((images, question, system_prompt, dict(headers or {})))
            yield "The object is "
            raise RuntimeError("stream disconnected")

    images = ImageRegistry()
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, _PartialFailureVlm()),
    )

    chunks = [
        chunk.text
        async for chunk in vision.stream(
            ImageQueryRequest(
                image=images.put(b"frame"),
                query="What is shown?",
            ),
        )
    ]

    assert chunks == ["The object is "]


async def test_streaming_image_query_reports_failure_before_any_output() -> None:
    class _ImmediateFailureVlm:
        async def stream_images(self, *_args, **_kwargs):
            if False:
                yield ""
            raise RuntimeError("stream unavailable")

    images = ImageRegistry()
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, _ImmediateFailureVlm()),
    )

    chunks = [
        chunk.text
        async for chunk in vision.stream(
            ImageQueryRequest(
                image=images.put(b"frame"),
                query="What is shown?",
            ),
        )
    ]

    assert chunks == ["VLM server unavailable — please retry."]


async def test_current_frame_tool_propagates_conversion_errors(monkeypatch) -> None:
    endpoint = _LiveEndpoint()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=ImageRegistry(),
    )
    await endpoint.frame_callback(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=time.time_ns() // 1_000,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data_sz=12,
            participant_id="alice",
            track_id="camera",
        )
    )

    def malformed_frame(_frame):
        raise ValueError("malformed pixels")

    monkeypatch.setattr(current_frame_module, "frame_to_pil", malformed_frame)

    with pytest.raises(RuntimeError, match="malformed pixels"):
        await frames.execute(CurrentFrameRequest(participant_id="alice"))


async def test_sample_runtime_streams_vision_through_voice_agent(monkeypatch) -> None:
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=images,
    )
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )

    class Session:
        def __init__(self) -> None:
            self.text = ""
            self.complete = asyncio.Event()
            self.started = asyncio.Event()
            self.text_topic = "agent.response"

        async def __aenter__(self):
            return self

        async def run(self, _handler, **_options) -> None:
            self.started.set()
            await asyncio.Event().wait()

        async def enqueue_response(
            self,
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            interrupt: bool = False,
            pts_us: int | None = None,
        ) -> None:
            assert participant_id == "alice"
            assert interrupt is True
            assert pts_us == 123

            async def consume() -> None:
                self.text = response if isinstance(response, str) else "".join([chunk async for chunk in response])
                self.complete.set()

            asyncio.create_task(consume())

        async def close(self) -> None:
            pass

    session = Session()
    monkeypatch.setattr(
        voice_runtime_module,
        "_VoiceSession",
        lambda **_kwargs: session,
    )
    runtime = AgentRuntime()
    runtime.register(
        "simple-vlm",
        SimpleVlmAgent(lambda: (frames, vision), endpoint.set_status),
    )
    runtime.register(
        "voice",
        VoiceAgent(
            query_topic=USER_QUERY_TOPIC,
            stt=object(),
            tts=object(),
            vad=object(),
            voice_gate=object(),
            text_input=False,
        ),  # type: ignore[arg-type]
    )
    assert endpoint.frame_callback is not None
    await endpoint.frame_callback(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=time.time_ns() // 1_000,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data_sz=12,
            participant_id="alice",
            track_id="camera",
        )
    )
    events = []
    subscriber = "simple-vlm-agent-vision"
    intercept = "simple-vlm-agent-vision-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-9"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        async with runtime:
            await runtime.publish(
                USER_QUERY_TOPIC,
                UserQuery(text="What is shown?", timestamp_us=123),
                participant_id="alice",
                source="test-input",
            )
            await asyncio.wait_for(session.complete.wait(), 1.0)
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert session.text == "a blue square"
    images, question, system_prompt, headers = vlm.calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-9"
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    starts = {
        event.name: event.to_dict()
        for event in events
        if event.kind == "scope" and event.to_dict().get("scope_category") == "start"
    }
    assert starts["simple-vlm.turn"]["parent_uuid"] != (starts["agent:simple-vlm"]["uuid"])
    tool_starts = [
        event.to_dict()
        for event in events
        if event.kind == "scope"
        and event.to_dict().get("scope_category") == "start"
        and event.to_dict().get("category") == "tool"
    ]
    assert tool_starts
    assert tool_starts[0]["parent_uuid"] == starts["simple-vlm.turn"]["uuid"]
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_simple_vlm_agent_handles_global_interruption_event() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingVision:
        async def stream(self, _request):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield SimpleNamespace(text="unreachable")

    runtime = AgentRuntime()
    runtime.register(
        "simple-vlm",
        SimpleVlmAgent(
            lambda: (_SelectedFrameTool(), BlockingVision()),  # type: ignore[return-value]
            _ignore_status,
        ),
    )

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="What is shown?", timestamp_us=123),
            participant_id="alice",
            source="test-input",
        )
        await asyncio.wait_for(started.wait(), 1.0)
        await runtime.publish(
            INTERRUPTED_TOPIC,
            VoiceInterrupted(),
            source="voice.interruption",
        )
        await asyncio.wait_for(cancelled.wait(), 1.0)
