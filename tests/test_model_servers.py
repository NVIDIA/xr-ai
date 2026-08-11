# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared, profile-driven model-server stack."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PATH = _REPO_ROOT / "agent-samples/model-servers/main.py"
_SPEC = importlib.util.spec_from_file_location("model_servers_main", _MAIN_PATH)
assert _SPEC and _SPEC.loader
_model_servers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_model_servers)

_OMNI_PATH = (
    _REPO_ROOT
    / "services/nemotron-omni-llm/nemotron_omni_llm_server/__main__.py"
)
_OMNI_SPEC = importlib.util.spec_from_file_location("nemotron_omni_main", _OMNI_PATH)
assert _OMNI_SPEC and _OMNI_SPEC.loader
_omni = importlib.util.module_from_spec(_OMNI_SPEC)
_OMNI_SPEC.loader.exec_module(_omni)

_EMBEDDING_PATH = (
    _REPO_ROOT / "services/embedding-server/embedding_server/__main__.py"
)
_EMBEDDING_SPEC = importlib.util.spec_from_file_location(
    "embedding_server_main", _EMBEDDING_PATH
)
assert _EMBEDDING_SPEC and _EMBEDDING_SPEC.loader
_embedding = importlib.util.module_from_spec(_EMBEDDING_SPEC)
_EMBEDDING_SPEC.loader.exec_module(_embedding)


def test_default_profile_uses_nano_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "spark")

    processes, credentials = _model_servers._build_processes("vlm_llm")

    assert [process.name for process in processes] == ["stt", "agent-llm", "vlm", "embedding"]
    assert [process.port for process in processes] == [8103, 8107, 8100, 8109]
    assert credentials == ()


def test_omni_profile_replaces_nano_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, _ = _model_servers._build_processes("omni")

    assert [process.name for process in processes] == ["stt", "omni", "embedding"]
    assert [process.port for process in processes] == [8103, 8108, 8109]
    assert str(processes[1].config) == "yaml/dual_48G_ada/nemotron_omni_llm_server.yaml"


def test_nim_profile_mixes_nim_containers_and_local_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, credentials = _model_servers._build_processes("vlm_llm_nim")

    assert [process.name for process in processes] == [
        "llm-nim", "vlm-nim", "stt", "embedding",
    ]
    assert [process.port for process in processes] == [8106, 8100, 8103, 8109]
    assert credentials == ("NGC_API_KEY",)


def test_vlm_speech_nim_profile_serves_speech_from_riva_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, credentials = _model_servers._build_processes("vlm_speech_nim")

    assert [process.name for process in processes] == [
        "stt-nim", "tts-nim", "vlm-nim", "embedding",
    ]
    assert credentials == ("NGC_API_KEY",)


@pytest.mark.parametrize(
    ("selection", "service", "config_name", "gpu"),
    [
        ("vlm_llm", "embedding", "embedding_server.yaml", "0"),
        ("omni", "embedding", "embedding_server_omni.yaml", "1"),
        ("vlm_llm_nim", "embedding", "embedding_server.yaml", "0"),
        ("vlm_llm_nim", "stt", "stt_server.yaml", "1"),
        ("vlm_llm_nim", "vlm-nim", "nim_vlm_server.yaml", "0"),
        ("vlm_speech_nim", "vlm-nim", "nim_vlm_server_vlm_speech_nim.yaml", "1"),
    ],
)
def test_dual_ada_configs_follow_profile_gpu_layout(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    service: str,
    config_name: str,
    gpu: str,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, _ = _model_servers._build_processes(selection)
    process = next(p for p in processes if p.name == service)
    config_path = _REPO_ROOT / "agent-samples/model-servers" / str(process.config)

    assert config_path.name == config_name
    assert yaml.safe_load(config_path.read_text())["cuda_visible_devices"] == gpu


@pytest.mark.parametrize("profile", ["96G_blackwell", "dual_48G_ada", "spark"])
def test_omni_profiles_select_supported_vllm_images(profile: str) -> None:
    profile_path = (
        _REPO_ROOT
        / "agent-samples/model-servers/yaml"
        / profile
        / "nemotron_omni_llm_server.yaml"
    )

    config = yaml.safe_load(profile_path.read_text())

    assert config["vllm_backend"] == "docker"
    assert config["vllm_image"] == "vllm/vllm-openai:v0.20.0"
    assert config["extra_pip"] == []
    if profile == "96G_blackwell":
        assert config["moe_backend"] == "triton"
    else:
        assert "moe_backend" not in config


def test_stop_cleans_every_service(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: stopped.extend(services) or True,
    )

    _model_servers._stop_models()

    assert set(stopped) == {
        ("stt-nim", 9010),
        ("tts-nim", 9011),
        ("llm-nim", 8106),
        ("vlm-nim", 8100),
        ("stt", 8103),
        ("agent-llm", 8107),
        ("omni", 8108),
        ("vlm", 8100),
        ("embedding", 8109),
    }


@pytest.mark.parametrize(
    ("selection", "expected_stopped_ports"),
    [
        # The selected profile's ports are kept; everything else is stopped.
        ("omni", {9010, 9011, 8106, 8100, 8107}),
        ("vlm_llm", {9010, 9011, 8106, 8108}),
        ("vlm_llm_nim", {9010, 9011, 8107, 8108}),
    ],
)
def test_starting_profile_stops_unselected_services(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected_stopped_ports: set[int],
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: stopped.extend(services) or True,
    )

    processes, _ = _model_servers._build_processes(selection)
    _model_servers._stop_unselected_services(processes)

    assert {port for _, port in stopped} == expected_stopped_ports


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "vlm_llm"),
        (["--vlm-llm-stack"], "vlm_llm"),
        (["--omni-stack"], "omni"),
        (["--models", "vlm_llm_nim"], "vlm_llm_nim"),
    ],
)
def test_cli_selects_requested_profile(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: str,
) -> None:
    selected: list[str] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_unselected_services", lambda _p: None)
    monkeypatch.setattr(
        _model_servers, "_build_processes",
        lambda selection: (selected.append(selection) or [], ()),
    )
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["model_servers", *argv])

    _model_servers.run()

    assert selected == [expected]


def test_build_processes_rejects_unknown_services(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")
    profile = tmp_path / "models.custom.json"
    profile.write_text(json.dumps({"models": {"vision": {
        "adapter": {"preset": "cosmos_vlm"},
        "endpoint": {"base_url": "http://localhost:8100"},
        "deployment": {"ownership": "managed", "service": "no-such-service"},
    }}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown services"):
        _model_servers._build_processes(str(profile))


def test_profile_path_argument_loads_custom_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")
    profile = tmp_path / "models.custom.json"
    profile.write_text(json.dumps({"models": {"vision": {
        "adapter": {"preset": "cosmos_vlm"},
        "endpoint": {"base_url": "http://localhost:8100"},
        "deployment": {"ownership": "managed", "service": "vlm"},
    }}}), encoding="utf-8")

    processes, _ = _model_servers._build_processes(str(profile))

    assert [process.name for process in processes] == ["vlm"]
    # Config variants key off the profile filename stem; a custom name has
    # no variants and falls back to the service defaults.
    assert str(processes[0].config) == "yaml/dual_48G_ada/vlm_server.yaml"


def test_cli_requires_profile_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    required: list[str] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _model_servers, "require_credentials",
        lambda name, **kw: required.append(name),
    )
    monkeypatch.setattr(_model_servers, "_stop_unselected_services", lambda _p: None)
    monkeypatch.setattr(
        _model_servers, "_build_processes",
        lambda _selection: ([], ("NGC_API_KEY",)),
    )
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["model_servers", "--models", "vlm_llm_nim"])

    _model_servers.run()

    assert required == ["HF_TOKEN", "NGC_API_KEY"]


def test_cli_aborts_when_unselected_services_cannot_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "stop_persistent_servers", lambda _services: False)
    monkeypatch.setattr(
        _model_servers, "_build_processes", lambda _selection: ([], ()),
    )
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: started.append(True))
    monkeypatch.setattr(sys, "argv", ["model_servers", "--omni-stack"])

    with pytest.raises(RuntimeError, match="could not stop persistent servers"):
        _model_servers.run()

    assert started == []


def test_omni_forwards_configured_moe_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(_omni, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _omni,
        "load_config",
        lambda: ({"moe_backend": "triton"}, Path("."), None),
    )
    monkeypatch.setattr(_omni, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_omni, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_omni, "gpu_compute_major", lambda: 10)
    monkeypatch.setattr(_omni, "serve", lambda **kwargs: captured.update(kwargs))

    _omni.run()

    args = captured["extra_serve_args"]
    assert args[args.index("--moe-backend") + 1] == "triton"


def test_embedding_default_cache_tracks_service_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[Path] = []
    project = _REPO_ROOT / "services/embedding-server"

    def resolve_cache(_cfg: dict, yaml_dir: Path, *, default: str) -> Path:
        path = (yaml_dir / default).resolve()
        resolved.append(path)
        return path

    monkeypatch.setattr(_embedding, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "load_config", lambda: ({}, project, None))
    monkeypatch.setattr(_embedding, "resolve_model_cache", resolve_cache)
    monkeypatch.setattr(_embedding, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "serve", lambda **_kwargs: None)

    _embedding.run()

    assert resolved == [_REPO_ROOT / "models"]


def test_stop_needs_no_stack_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_models", lambda: calls.append("stop"))
    monkeypatch.setattr(sys, "argv", ["model_servers", "--stop"])

    _model_servers.run()

    assert calls == ["stop"]
