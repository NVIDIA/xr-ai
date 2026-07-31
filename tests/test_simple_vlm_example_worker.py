# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the packaged simple-vlm-example worker."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_nat.functions.vision import StreamingVisionConfig
from xr_ai_voice import VoiceQuery, VoiceSession
from xr_ai_voicegate import VoiceGateConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "simple-vlm-example"
_WORKER_DIR = _SAMPLE_DIR / "worker"
sys.path.insert(0, str(_WORKER_DIR))

from simple_vlm_example_worker import __main__ as worker_main  # noqa: E402
from simple_vlm_example_worker import app  # noqa: E402
from simple_vlm_example_worker.config import load_config  # noqa: E402


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


class _VisionFunction:
    def __init__(self) -> None:
        self.requests = []

    async def astream(self, request):
        self.requests.append(request)
        for text in ("a ", "blue square"):
            yield SimpleNamespace(text=text)


class _VisionConfig:
    instances: list["_VisionConfig"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.released: list[str] = []
        self.instances.append(self)

    def release(self, participant_id: str) -> None:
        self.released.append(participant_id)


class _Builder:
    def __init__(self, function: _VisionFunction) -> None:
        self.function = function
        self.added: list[tuple[str, object]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def add_function(self, name: str, config: object):
        self.added.append((name, config))
        return self.function


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
    assert "xr-ai-nat[vision,voice]" in dependencies
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
    config = load_config(_SAMPLE_DIR / "yaml" / "simple_vlm_example_worker.yaml")
    prompt = (
        _WORKER_DIR
        / "simple_vlm_example_worker"
        / "prompts"
        / "system.txt"
    ).read_text()

    assert config.model_backend == "local"
    assert config.models_yaml == _SAMPLE_DIR / "yaml" / "models.yaml"
    assert config.voice_gate_yaml == _SAMPLE_DIR / "yaml" / "voice_gate.yaml"
    assert config.system_prompt == prompt
    assert config.default_prompt == "Describe what you see."
    assert config.frame_max_age_s == 5.0
    assert config.frame_timeout_s == 5.0
    assert config.idle_timeout_secs is None


def test_config_keeps_nim_overlay_and_inline_prompt_compatibility(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "model_backend: NIM\n"
        "models_yaml: ignored.yaml\n"
        "voice_gate_yaml: gate.yaml\n"
        "system_prompt: custom prompt\n"
        "idle_timeout_secs: 30\n"
    )

    config = load_config(config_path)

    assert config.model_backend == "nim"
    assert config.models_yaml == tmp_path / "models.nim.yaml"
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
    function = _VisionFunction()
    builder = _Builder(function)
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
    monkeypatch.setattr(app, "WorkflowBuilder", lambda: builder)
    monkeypatch.setattr(app, "StreamingVisionConfig", _VisionConfig)

    def make_session(**kwargs):
        session = VoiceSession(transport=transport, **kwargs)  # type: ignore[arg-type]
        sessions.append(session)

        async def run(handler, **options) -> None:
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
    _VisionConfig.instances.clear()

    await app.run_app(config, ready_file=ready_file)

    assert ready_file.exists()
    assert stt.health_calls == tts.health_calls == vlm.health_calls == 1
    assert stt.close_calls == tts.close_calls == vlm.close_calls == 1
    assert transport.shutdown_calls == 1
    assert sessions[0].text_topic == "vlm.response"
    assert builder.added == [("perception", _VisionConfig.instances[0])]
    assert _VisionConfig.instances[0].kwargs["endpoint"] is transport.endpoint
    assert _VisionConfig.instances[0].released == ["alice"]
    assert function.requests[0].participant_id == "alice"
    assert function.requests[0].query == "What is in front of me?"
    assert streamed == ["a ", "blue square"]
    assert run_options["interrupt_on_supersede"] is True
    assert text_inputs[0]["session"] is sessions[0]
    assert text_inputs[0]["fresh_match"] is True
    assert text_inputs[0]["transform"]("PING") == config.default_prompt
    assert text_inputs[0]["transform"]("What is this?") == "What is this?"


async def test_sample_handler_streams_a_live_frame_question() -> None:
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    vision_config = StreamingVisionConfig(
        endpoint=endpoint,
        vlm=vlm,
        system_prompt="Answer briefly.",
    )

    async with WorkflowBuilder() as builder:
        vision = await builder.add_function("perception", vision_config)
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
        response = await handler(
            VoiceQuery(
                participant_id="alice",
                text="What is shown?",
                fresh_match=True,
                timestamp_us=123,
            )
        )
        tokens = [token async for token in response]

    assert tokens == ["a ", "blue ", "square"]
    image, question, system_prompt = vlm.calls[0]
    assert image.startswith("data:image/jpeg;base64,")
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
