# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native tea-making sample."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import nemo_relay
import pytest
import yaml
from xr_ai_interaction import ParticipantJoined, ParticipantLeft
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))
from tea_making_worker.application.application_config import load_config  # noqa: E402
from tea_making_worker.application.tea_application import _CLIENT_TEXT_TOPIC, _relay_event_log  # noqa: E402
from tea_making_worker.application.tea_event_viewer import TeaWebEventsAgent  # noqa: E402
from tea_making_worker.application.tea_events import (  # noqa: E402
    GUIDANCE_NOTICE_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    GuidanceNotice,
    GuidanceRecord,
)


def _load_main():
    path = _SAMPLE / "main.py"
    spec = importlib.util.spec_from_file_location("tea_making_sample_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_omni_supplies_both_language_and_vision() -> None:
    models = json.loads((_SAMPLE / "yaml/models.local.json").read_text())["models"]
    assert models["llm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["adapter"]["capabilities"]["vision"] is True
    assert models["vlm"]["adapter"]["reasoning_field"] == "reasoning_content"
    assert all(("deployment" not in model for model in models.values()))
    assert "cosmos" not in json.dumps(models).lower()


def test_hybrid_speaker_stt_configuration_matches_model_stack() -> None:
    """Conversation controls and one-off addressing share the speaker-ASR stack."""
    from xr_ai_voicegate import load_voice_gate_config

    gate = load_voice_gate_config(_SAMPLE / "yaml/voice_gate.yaml")
    gate_yaml = yaml.safe_load((_SAMPLE / "yaml/voice_gate.yaml").read_text())
    assert gate_yaml["mode"] == "hybrid"
    assert gate.magic_phrases == ("assistant",)
    assert gate_yaml["conversation"]["start_phrase"] == "Start talking to me"
    assert gate_yaml["conversation"]["stop_phrase"] == "Stop talking to me"
    assert gate_yaml["speaker"]["enabled"] is True
    assert gate_yaml["speaker"]["backend"] == "required"
    models = json.loads((_SAMPLE / "yaml/models.local.json").read_text())["models"]
    assert "stt" not in models

    spec = importlib.util.spec_from_file_location("tea_model_servers", _SAMPLE / "model-servers/main.py")
    assert spec and spec.loader
    servers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(servers)
    assert "speaker-stt" in servers._SERVICES and "stt" not in servers._SERVICES
    for profile_path in (_SAMPLE / "model-servers/yaml").glob("*.yaml"):
        profile = yaml.safe_load(profile_path.read_text())
        assert "speaker-stt" in profile and "stt" not in profile


def test_materialized_config_stays_inside_runtime_dir(tmp_path: Path) -> None:
    sample_main = _load_main()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    assert sample_main._materialize_worker_config(runtime_dir) == runtime_dir / "tea_making_worker.yaml"
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "tea_making_worker.yaml").symlink_to(tmp_path / "elsewhere.yaml")
    with pytest.raises(ValueError, match="is outside"):
        sample_main._materialize_worker_config(escaped)


def test_launcher_declares_only_sample_owned_processes(tmp_path: Path) -> None:
    sample_main = _load_main()
    worker_config = sample_main._materialize_worker_config(tmp_path)
    processes = sample_main._build_processes(worker_config)
    names = [process.name for process in processes]
    assert names == ["hub", "rag", "worker"]


def test_launcher_only_exposes_web_events_override() -> None:
    sample_main = _load_main()
    default = sample_main._parse_args([])
    exposed = sample_main._parse_args(["--expose-web-events"])
    assert default.expose_web_events is False
    assert exposed.expose_web_events is True
    assert _CLIENT_TEXT_TOPIC == "agent.response"


def test_web_event_config_defaults_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    assert config.web_events_host == "127.0.0.1"
    assert config.web_events_port == 8093
    assert config.web_events_max_events == 5000
    invalid = tmp_path / "invalid-web-events.yaml"
    invalid.write_text("web_events_host: ''\n")
    with pytest.raises(ValueError, match="web_events_host"):
        load_config(invalid)
    invalid.write_text("web_events_port: 70000\n")
    with pytest.raises(ValueError, match="web_events_port"):
        load_config(invalid)
    invalid.write_text("web_events_max_events: 0\n")
    with pytest.raises(ValueError, match="web_events_max_events"):
        load_config(invalid)
    sample_main = _load_main()
    exposed_config = sample_main._materialize_worker_config(tmp_path, expose_web_events=True)
    assert load_config(exposed_config).web_events_host == "0.0.0.0"
    assert load_config(exposed_config).voice_gate_yaml == _SAMPLE / "yaml/voice_gate.yaml"
    custom_gate = tmp_path / "custom-voice.yaml"
    custom_gate.write_text("mode: hybrid\nmagic_phrases: [hey agent]\n", encoding="utf-8")
    custom_source = tmp_path / "custom-worker.yaml"
    custom_source.write_text(
        (_SAMPLE / "yaml/tea_making_worker.yaml").read_text().replace("voice_gate.yaml", "custom-voice.yaml"),
        encoding="utf-8",
    )
    monkeypatch.setattr(sample_main, "_WORKER_CONFIG", custom_source)
    custom_config = sample_main._materialize_worker_config(tmp_path)
    assert load_config(custom_config).voice_gate_yaml == custom_gate


@pytest.mark.asyncio
async def test_typed_events_are_projected_to_web_topics() -> None:
    captured: list[tuple[WebEvent, str | None]] = []

    class Capture(Agent):
        def __init__(self) -> None:
            super().__init__()

        @subscribe(WEB_EVENT_TOPIC)
        async def event(self, event: WebEvent, ctx: RuntimeContext) -> None:
            captured.append((event, ctx.metadata.participant_id))

    runtime = AgentRuntime()
    runtime.register("web-event-adapter", TeaWebEventsAgent())
    runtime.register("capture", Capture())
    participant_id = "participant-web"
    publications = (
        (GUIDANCE_RECORD_TOPIC, GuidanceRecord(timestamp_us=2, event="step.enter", message="Fill the kettle.")),
        (GUIDANCE_NOTICE_TOPIC, GuidanceNotice(timestamp_us=3, text="The water is ready.")),
        (PARTICIPANT_JOINED_TOPIC, ParticipantJoined()),
        (PARTICIPANT_LEFT_TOPIC, ParticipantLeft()),
    )
    async with runtime:
        for topic, event in publications:
            await runtime.publish(topic, event, participant_id=participant_id)
    assert [event.topic for event, _participant in captured] == [
        "guidance.events",
        "guidance.notices",
        "participant.lifecycle",
        "participant.lifecycle",
    ]
    assert {participant for _event, participant in captured} == {participant_id}
    assert captured[-2][0].payload == {"event": "joined"}
    assert captured[-1][0].payload == {"event": "left"}


@pytest.mark.asyncio
async def test_web_viewer_surrounds_runtime_lifecycle() -> None:
    runtime = AgentRuntime()
    viewer = runtime.register("web-events", WebEventsAgent(host="127.0.0.1", port=0, max_events=8))
    runtime.register("web-event-adapter", TeaWebEventsAgent())
    assert not viewer.running
    async with viewer:
        assert viewer.running
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="participant-live")
        assert viewer.running
    assert not viewer.running


@pytest.mark.asyncio
async def test_relay_event_log_buffers_and_excludes_stream_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = None

    def register(_name, registered_callback):
        nonlocal callback
        callback = registered_callback

    async def flush_async() -> None:
        return None

    monkeypatch.setattr(nemo_relay.subscribers, "register", register)
    monkeypatch.setattr(nemo_relay.subscribers, "flush_async", flush_async)
    monkeypatch.setattr(nemo_relay.subscribers, "deregister", lambda _name: None)

    class Event:
        def __init__(self, name: str) -> None:
            self.kind = "mark"
            self.name = name

        def to_json(self) -> str:
            return json.dumps({"name": self.name})

    async with _relay_event_log(tmp_path):
        assert callback is not None
        callback(Event("llm.chunk"))
        callback(Event("turn.summary"))
    events = [yaml.safe_load(line) for line in (tmp_path / "relay-events.jsonl").read_text().splitlines()]
    names = [event["name"] for event in events]
    assert "llm.chunk" not in names
    assert "turn.summary" in names


def test_default_prompt_policies_come_from_main_worker_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    config = load_config(None)
    names = (
        "conversation_system_prompt",
        "current_view_prompt",
    )
    for name in names:
        assert getattr(config, name) == raw[name].strip()
    assert config.sop.advancement == "automatic"
    assert config.models_config == _SAMPLE / "yaml/models.local.json"
    assert not (_WORKER / "tea_making_worker/prompts").exists()
    raw["conversation_system_prompt"] = "Explicit override"
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(raw))
    override = load_config(path)
    assert override.conversation_system_prompt == "Explicit override"
    assert override.current_view_prompt == config.current_view_prompt


@pytest.mark.parametrize(
    "name",
    [
        "conversation_system_prompt",
        "current_view_prompt",
    ],
)
def test_selected_yaml_has_no_hidden_prompt_or_decision_fallback(tmp_path, name):
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    del raw[name]
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=name):
        load_config(path)


@pytest.mark.parametrize("name", ["conversation_system_prompt", "current_view_prompt"])
def test_prompt_file_options_are_replaced_by_inline_yaml(tmp_path, name):
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    raw[name + "_file"] = "legacy.txt"
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=name + "_file"):
        load_config(path)


def test_installed_worker_default_uses_sample_yaml_or_requires_explicit_config(tmp_path, monkeypatch):
    import tea_making_worker.application.application_config as config_module

    monkeypatch.setattr(config_module, "_WORKER_PACKAGE", tmp_path / "site-packages/tea_making_worker")
    monkeypatch.chdir(_SAMPLE)
    assert load_config(None).sop.advancement == "automatic"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="--config"):
        load_config(None)
    explicit = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    assert explicit.current_view_prompt


def test_reusable_components_have_no_application_dependencies():
    component_directory = _WORKER / "tea_making_worker/components"
    for path in component_directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, f"{path.name} imports outside components"
                imported_names = [node.module or ""]
            elif isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            else:
                continue
            assert all(
                not name.startswith("tea_making_worker") or name.startswith("tea_making_worker.components")
                for name in imported_names
            ), f"{path.name} imports application through the package root"


def test_fresh_component_import_does_not_load_the_application():
    # A clean interpreter catches eager package-root reexports that a static
    # dependency check misses, without importing/starting the application.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tea_making_worker.components import run_visual_verification_flow, "
        "run_vision_rag_retrieval_flow, run_timer_flow; "
        "assert not any(name.startswith('tea_making_worker.application') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script, str(_WORKER)], check=True, capture_output=True, text=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_disposal", (False, True))
async def test_application_drains_sop_work_before_closing_borrowed_services(monkeypatch, tmp_path, fail_disposal):
    """Voice input may stop first; models/images outlive actual owned inference."""
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from tea_making_worker.application import tea_application as application
    from tea_making_worker.application.tea_events import USER_QUERY_TOPIC
    from tea_making_worker.application.tea_sop_agent import SOPAgent
    from xr_ai_interaction import UserQuery
    from xr_ai_models import DecisionAnswer, DecisionResponse
    from xr_ai_tools.image import ImageRegistry

    inference_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    images = ImageRegistry()
    instances = []
    endpoint = Mock(closed=False, set_status=AsyncMock())

    class Client:
        def __init__(self, name):
            self.name = name
            self.close_count = 0
            self.health = AsyncMock(return_value=True)

        async def close(self):
            if self.name != "speech" and inference_started.is_set():
                assert cleanup_finished.is_set(), "borrowed service closed before inference cleanup"
            self.close_count += 1

        async def decide(self, _state, questions, **_kwargs):
            choices = {"operation": "start", "referent": "workflow"}
            return DecisionResponse(
                {
                    name: DecisionAnswer(
                        choices[name],
                        1.0,
                        {label: float(label == choices[name]) for label in question.criteria},
                    )
                    for name, question in questions.items()
                },
                "fixture",
                {},
                {},
            )

    # Deliberate alias verifies that shared client ownership is based on object
    # identity rather than one close for every configured model role.
    language_and_decisions = Client("language-and-decisions")
    vision = Client("vision")
    retrieval = Client("retrieval")
    speech = Client("speech")

    class ScopedSOP(SOPAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

        async def participant_left(self, participant_id):
            if fail_disposal and not finish_cleanup.is_set():
                raise RuntimeError("scope disposal blocked")
            await super().participant_left(participant_id)

        async def _procedure(self, session):
            async def infer():
                image = images.put(b"owned immutable frame", owner=session.participant_id)
                inference_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await finish_cleanup.wait()
                    assert endpoint.closed
                    assert images.resolve(image) == b"owned immutable frame"
                    cleanup_finished.set()

            await self._run_step(session, 0, infer)

    class Voice(Agent):
        def __init__(self, kwargs):
            super().__init__()
            self.kwargs = kwargs

        async def run(self, runtime):
            try:
                await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p")
                await runtime.publish(
                    USER_QUERY_TOPIC,
                    UserQuery(text="start tea making", timestamp_us=100),
                    participant_id="p",
                )
                await inference_started.wait()
            finally:
                endpoint.closed = True
                for client in (speech, *self.kwargs.get("closeables", ())):
                    await client.close()

    monkeypatch.setattr(application, "make_llm", lambda *_: language_and_decisions)
    monkeypatch.setattr(application, "make_decision", lambda *_: language_and_decisions)
    monkeypatch.setattr(application, "make_vlm", lambda *_: vision)
    monkeypatch.setattr(application, "make_tts", lambda *_: speech)
    monkeypatch.setattr(application, "RAGTools", lambda *_: retrieval)
    monkeypatch.setattr(application, "HubVoiceTransport", lambda: SimpleNamespace(endpoint=endpoint))
    monkeypatch.setattr(application, "VoiceAgent", SimpleNamespace(from_models=lambda **kwargs: Voice(kwargs)))
    monkeypatch.setattr(application, "ImageRegistry", lambda: images)
    monkeypatch.setattr(application, "SOPAgent", ScopedSOP)
    config = replace(load_config(None), artifacts_dir=tmp_path, web_events_port=0)
    task = asyncio.create_task(application.run_app(config))
    clients = (language_and_decisions, vision, retrieval)
    try:
        start_wait = asyncio.create_task(inference_started.wait())
        done, _pending = await asyncio.wait((task, start_wait), timeout=2, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
        assert start_wait in done, "application did not enter inference"
        if fail_disposal:
            with pytest.raises(ExceptionGroup) as failure:
                await task
            assert "scope disposal blocked" in repr(failure.value)
            assert instances[0]._sessions["p"].procedure is not None
            assert not instances[0]._sessions["p"].procedure.done()
            assert len(images) == 1
        else:
            await asyncio.wait_for(cleanup_started.wait(), 2)
            assert not task.done() and len(images) == 1
        assert all(client.close_count == 0 for client in clients)
        finish_cleanup.set()
        if fail_disposal:
            await instances[0].stop()
            assert cleanup_finished.is_set()
            # Failed application cleanup intentionally retained borrowed services;
            # the test now disposes its own remaining fixture resources.
            for client in clients:
                await client.close()
            images.clear()
        else:
            await asyncio.wait_for(task, 2)
            assert len(images) == 0
        assert all(client.close_count == 1 for client in clients)
        assert speech.close_count == 1
    finally:
        finish_cleanup.set()
        start_wait.cancel()
        await asyncio.gather(start_wait, return_exceptions=True)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for instance in instances:
            await instance.stop()
            await instance._context.close()
