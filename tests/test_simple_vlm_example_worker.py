# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the packaged simple-vlm-example worker."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import nemo_relay
import pytest
import tomllib
import yaml
from xr_ai_hub import FrameData, FrameSignal, PixelFormat, ProcessorEndpoint
from xr_ai_models import VLMService
from xr_ai_voice import VoiceQuery, VoiceSession
from xr_ai_voicegate import VoiceGateConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "simple-vlm-example"
_WORKER_DIR = _SAMPLE_DIR / "worker"
sys.path.insert(0, str(_WORKER_DIR))

from simple_vlm_example_worker import __main__ as worker_main  # noqa: E402  # pyright: ignore[reportMissingImports]
from simple_vlm_example_worker import app  # noqa: E402  # pyright: ignore[reportMissingImports]
from simple_vlm_example_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from xr_ai_nat.live_vision import LiveVisionTool  # noqa: E402


class _Service:
    def __init__(self) -> None:
        self.health_calls = 0
        self.close_calls = 0

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def close(self) -> None:
        self.close_calls += 1


class _Transport:
    def __init__(self) -> None:
        self.endpoint = object()
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _LiveVisionTool:
    instances: list["_LiveVisionTool"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.released: list[str] = []
        self.instances.append(self)

    async def stream(self, request):
        self.requests.append(request)
        for text in ("a ", "blue square"):
            yield SimpleNamespace(text=text)

    def release(self, participant_id: str) -> None:
        self.released.append(participant_id)


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

    async def stream(self, image, question: str, *, system_prompt: str = ""):
        self.calls.append((image, question, system_prompt))
        for token in ("a ", "blue ", "square"):
            yield token


def test_worker_is_a_package_with_module_and_console_entry_points() -> None:
    project = tomllib.loads((_WORKER_DIR / "pyproject.toml").read_text())
    package = _WORKER_DIR / "simple_vlm_example_worker"
    dependencies = set(project["project"]["dependencies"])

    assert project["project"]["scripts"]["simple_vlm_example_worker"] == (
        "simple_vlm_example_worker.__main__:run"
    )
    assert "xr-ai-hub-client" in dependencies
    assert project["tool"]["uv"]["sources"]["xr-ai-hub-client"]["path"] == (
        "../../../agent-sdk/xr-ai-hub-client"
    )
    assert "xr-ai-nat[relay,live-vision]" in dependencies
    assert all("[vision" not in dependency and "[voice" not in dependency for dependency in dependencies)
    assert "xr-ai-voice" in dependencies
    assert "xr-ai-pipecat" not in dependencies
    assert all("mcp" not in dependency.lower() for dependency in dependencies)
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "simple_vlm_example_worker"
    ]
    assert {
        "__init__.py",
        "__main__.py",
        "app.py",
        "config.py",
        "prompts/system.txt",
    } <= {
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file()
    }
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
    prompt = (
        _WORKER_DIR
        / "simple_vlm_example_worker"
        / "prompts"
        / "system.txt"
    ).read_text()

    assert config.models_config == _SAMPLE_DIR / "yaml" / "models.local.json"
    assert config.voice_gate_yaml == _SAMPLE_DIR / "yaml" / "voice_gate.yaml"
    assert config.system_prompt == prompt
    assert "Speak directly to me in second person" in prompt
    assert 'Never refer to "the user" in the third person.' in prompt
    assert "when the user" not in prompt
    assert "If the user" not in prompt
    assert config.default_prompt == "Describe what you see."
    assert config.frame_max_age_s == 5.0
    assert config.frame_timeout_s == 5.0
    assert config.idle_timeout_secs is None
    assert "system_prompt_file" not in raw


def test_config_without_a_file_uses_packaged_defaults(tmp_path) -> None:
    prompt = (
        _WORKER_DIR
        / "simple_vlm_example_worker"
        / "prompts"
        / "system.txt"
    ).read_text()

    for config_path in (None, tmp_path / "missing.yaml"):
        config = load_config(config_path)
        assert config.system_prompt == prompt
        expected_parent = Path() if config_path is None else tmp_path
        assert config.models_config == expected_parent / "models.local.json"


def test_blank_inline_prompt_uses_packaged_default(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("system_prompt:\n")
    prompt = (
        _WORKER_DIR
        / "simple_vlm_example_worker"
        / "prompts"
        / "system.txt"
    ).read_text()

    assert load_config(config_path).system_prompt == prompt


def test_config_keeps_deployment_profile_and_inline_prompt_compatibility(
    tmp_path,
) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "models_config: models.hosted.json\n"
        "voice_gate_yaml: gate.yaml\n"
        "system_prompt: custom prompt\n"
        "idle_timeout_secs: 30\n"
    )

    config = load_config(config_path)

    assert config.models_config == tmp_path / "models.hosted.json"
    assert config.voice_gate_yaml == tmp_path / "gate.yaml"
    assert config.system_prompt == "custom prompt"
    assert config.idle_timeout_secs == 30.0


def test_config_rejects_a_non_mapping_yaml_document(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(config_path)


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
    text_inputs = []
    run_options = {}
    streamed = []

    monkeypatch.setattr(app, "setup_logging", lambda _name: None)
    monkeypatch.setattr(app, "load_models_config", lambda path: path)
    monkeypatch.setattr(app, "load_voice_gate_config", lambda _path: VoiceGateConfig())
    monkeypatch.setattr(app, "make_stt", lambda _models, _name: stt)
    monkeypatch.setattr(app, "make_vlm", lambda _models, _name: vlm)
    monkeypatch.setattr(app, "make_tts", lambda _models, _name: tts)
    monkeypatch.setattr(app, "LiveVisionTool", _LiveVisionTool)

    def make_session(**kwargs):
        session = VoiceSession(transport=transport, **kwargs)  # type: ignore[arg-type]
        sessions.append(session)

        async def run(handler, **options) -> None:
            if session.ready_file:
                session.ready_file.touch()
            run_options.update(options)
            response = await handler(
                VoiceQuery(
                    participant_id="alice",
                    text="What is in front of me?",
                    fresh_match=True,
                    timestamp_us=123,
                )
            )
            streamed.extend([chunk async for chunk in response])
            options["on_participant_left"]("alice")

        session.run = run  # type: ignore[method-assign]
        return session

    class CaptureTextInput:
        def __init__(self, **kwargs) -> None:
            text_inputs.append(kwargs)

    monkeypatch.setattr(app, "VoiceSession", make_session)
    monkeypatch.setattr(app, "TextMessageInput", CaptureTextInput)
    _LiveVisionTool.instances.clear()

    await app.run_app(config, ready_file=ready_file)

    assert ready_file.exists()
    assert stt.health_calls == tts.health_calls == vlm.health_calls == 1
    assert stt.close_calls == tts.close_calls == vlm.close_calls == 1
    assert transport.shutdown_calls == 1
    assert sessions[0].text_topic == "vlm.response"
    assert _LiveVisionTool.instances[0].kwargs["endpoint"] is transport.endpoint
    assert _LiveVisionTool.instances[0].kwargs["system_prompt"] == config.system_prompt
    assert _LiveVisionTool.instances[0].kwargs["frame_max_age_s"] == (
        config.frame_max_age_s
    )
    assert _LiveVisionTool.instances[0].kwargs["frame_timeout_s"] == (
        config.frame_timeout_s
    )
    assert _LiveVisionTool.instances[0].released == ["alice"]
    assert _LiveVisionTool.instances[0].requests[0].participant_id == "alice"
    assert _LiveVisionTool.instances[0].requests[0].query == "What is in front of me?"
    assert streamed == ["a ", "blue square"]
    assert run_options["interrupt_on_supersede"] is True
    assert text_inputs[0]["session"] is sessions[0]
    assert text_inputs[0]["fresh_match"] is True
    assert text_inputs[0]["transform"]("PING") == config.default_prompt
    assert text_inputs[0]["transform"]("What is this?") == "What is this?"


async def test_sample_handler_streams_a_live_frame_question() -> None:
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    vision = LiveVisionTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )

    handler = app._make_vision_handler(vision)
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
    subscriber = "simple-vlm-live-vision"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        response = await handler(
            VoiceQuery(
                participant_id="alice",
                text="What is shown?",
                fresh_match=True,
                timestamp_us=123,
            )
        )
        tokens = [token async for token in response]
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)

    assert tokens == ["a ", "blue ", "square"]
    image, question, system_prompt = vlm.calls[0]
    assert image.startswith("data:image/jpeg;base64,")
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert {"llm", "tool"} <= {getattr(event, "category", None) for event in events}
