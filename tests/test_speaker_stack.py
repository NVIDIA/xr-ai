# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker-STT stack selection and shared port cleanup; no GPU models."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from xr_ai_vllm import _docker

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "speaker_stack_main", _ROOT / "model-server-samples/model-servers/main.py",
)
assert _SPEC and _SPEC.loader
stack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stack)


def test_speaker_config_requires_http_port(tmp_path, monkeypatch):
    config = tmp_path / "yaml" / "custom" / "speaker_stt.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("host: 127.0.0.1\n")
    profile = tmp_path / "models.speaker.json"
    profile.write_text(json.dumps({
        "models": {}, "services": {"speaker-stt": {"ownership": "managed"}},
    }))
    monkeypatch.setattr(stack, "_BASE", tmp_path)
    with pytest.raises(ValueError, match="must declare port"):
        stack._build_processes(str(profile), "custom")


@pytest.mark.parametrize("hardware", ["dual_48G_ada", "96G_blackwell", "spark"])
def test_normal_profile_selects_speaker_asr_and_uses_hardware_profile(hardware):
    processes, credentials = stack._build_processes("default", hardware)
    names = [p.name for p in processes]
    assert "stt" not in names and "stt-nim" not in names
    assert names.count("speaker-stt") == 1
    speaker = next(p for p in processes if p.name == "speaker-stt")
    assert speaker.port == 8102 and speaker.launch_mode == "persist"
    assert speaker.command == "speaker_stt"
    cfg = yaml.safe_load(Path(speaker.config).read_text())
    assert cfg["host"] == "127.0.0.1"
    assert cfg["cuda_visible_devices"] == ("1" if hardware == "dual_48G_ada" else "0")
    assert (Path(speaker.config).parent / cfg["model_cache"]).resolve() == _ROOT / "models"
    assert credentials == ()


def test_custom_batch_profile_remains_supported(tmp_path):
    profile = tmp_path / "models.speech.json"
    profile.write_text(json.dumps({"models": {"stt": {
        "adapter": {"kind": "riva_grpc"}, "endpoint": {"base_url": "localhost:50051"},
        "deployment": {"ownership": "managed", "service": "stt-nim", "credentials": ["NGC_API_KEY"]},
    }}}))
    processes, credentials = stack._build_processes(str(profile), "dual_48G_ada")
    assert [p.name for p in processes] == ["stt-nim"]
    assert credentials == ("NGC_API_KEY",)


@pytest.mark.parametrize("declaration", [
    [], {"unknown": {"ownership": "managed"}}, {"speaker-stt": {"ownership": "reused"}},
    {"speaker-stt": {"ownership": "managed", "endpoint": "http://localhost:8102"}},
])
def test_invalid_operator_service_declaration_fails_before_launch(tmp_path, declaration):
    profile = tmp_path / "models.invalid.json"
    profile.write_text(json.dumps({"models": {}, "services": declaration}))
    with pytest.raises(ValueError, match="services"):
        stack._build_processes(str(profile), "dual_48G_ada")


def test_normal_cli_passes_only_profile_selection(monkeypatch):
    selected = []
    monkeypatch.setattr(stack, "setup_logging", lambda *_a, **_kw: None)
    monkeypatch.setattr(stack, "require_credentials", lambda *_a, **_kw: None)
    monkeypatch.setattr(stack, "_stop_unselected_services", lambda _p: None)
    monkeypatch.setattr(stack, "run_stack", lambda *_a, **_kw: None)
    monkeypatch.setattr(stack, "_build_processes", lambda *a, **kw: (selected.append((a, kw)) or [], ()))
    monkeypatch.setattr(sys, "argv", ["model_servers", "--models", "vlm_llm_nim"])
    stack.run()
    assert selected == [(("vlm_llm_nim", None), {})]


def test_profile_switches_stop_only_unselected_ports(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(stack, "stop_persistent_servers", lambda services: stopped.extend(services) or True)
    processes, _ = stack._build_processes("default", "dual_48G_ada")
    stack._stop_unselected_services(processes)
    assert ("speaker-stt", 8102) not in stopped
    assert {8103, 9010} <= {port for _, port in stopped}
    stopped.clear()
    profile = tmp_path / "models.batch.json"
    body = json.loads(stack._profile_path("default").read_text())
    body.pop("services")
    body["models"]["stt"]["deployment"]["ownership"] = "managed"
    profile.write_text(json.dumps(body))
    processes, _ = stack._build_processes(str(profile), "dual_48G_ada")
    stack._stop_unselected_services(processes)
    assert ("speaker-stt", 8102) in stopped
    assert ("stt", 8103) not in stopped
    stopped.clear()
    stack._stop_models()
    assert ("speaker-stt", 8102) in stopped


def test_speaker_cleanup_uses_configured_port_not_default(tmp_path, monkeypatch):
    config = tmp_path / "yaml" / "custom" / "speaker_stt.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("host: 127.0.0.1\nport: 12345\n")
    profile = tmp_path / "models.speaker.json"
    profile.write_text(json.dumps({
        "models": {}, "services": {"speaker-stt": {"ownership": "managed"}},
    }))
    monkeypatch.setattr(stack, "_BASE", tmp_path)
    processes, _ = stack._build_processes(str(profile), "custom")
    assert processes[0].port == 12345
    assert stack._known_service_ports() == [("speaker-stt", 12345)]


@pytest.mark.parametrize("environment, owned", [
    (b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8102\0", True),
    (b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8103\0", False),
    (b"PATH=/bin\0", False),
])
def test_speaker_cleanup_requires_matching_managed_listener(tmp_path, monkeypatch, environment, owned):
    proc = tmp_path / "proc" / "1234"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_text("python\0-m\0speaker_stt\0--_serve")
    (proc / "environ").write_bytes(environment)
    monkeypatch.setattr(_docker, "Path", lambda path: proc / path.rsplit("/", 1)[-1])
    assert _docker.is_xr_ai_server_process(1234, "speaker-stt", 8102) is owned
