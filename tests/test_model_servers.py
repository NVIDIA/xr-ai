# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared model-server stack selector."""
from __future__ import annotations

import importlib.util
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


def test_default_stack_uses_omni_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "spark")

    processes = _model_servers._build_processes()

    assert [process.name for process in processes] == ["stt", "omni", "vlm", "embedding"]
    assert [process.port for process in processes] == [8103, 8108, 8100, 8109]


def test_dual_ada_places_omni_opposite_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes = _model_servers._build_processes()
    configs = {
        process.name: yaml.safe_load(
            (_REPO_ROOT / "agent-samples/model-servers" / str(process.config)).read_text()
        )
        for process in processes
        if process.name in {"omni", "vlm"}
    }

    assert configs["omni"]["cuda_visible_devices"] == "1"
    assert configs["vlm"]["cuda_visible_devices"] == "0"


def test_dual_ada_embedding_stays_with_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    process = next(p for p in _model_servers._build_processes() if p.name == "embedding")
    config_path = _REPO_ROOT / "agent-samples/model-servers" / str(process.config)

    assert config_path.name == "embedding_server.yaml"
    assert yaml.safe_load(config_path.read_text())["cuda_visible_devices"] == "0"


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


def test_stop_cleans_every_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: stopped.extend(services) or True,
    )

    _model_servers._stop_models()

    assert stopped == [
        ("stt", 8103),
        ("agent-llm", 8107),
        ("vlm", 8100),
        ("omni", 8108),
        ("embedding", 8109),
    ]


def test_starting_stack_stops_only_replaced_nano(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: stopped.extend(services) or True,
    )

    _model_servers._stop_replaced_models()

    assert stopped == [("agent-llm", 8107)]


def test_cli_starts_single_default_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_replaced_models", lambda: calls.append("stop"))
    monkeypatch.setattr(_model_servers, "_build_processes", lambda: calls.append("build") or [])
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: calls.append("run"))
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    _model_servers.run()

    assert calls == ["stop", "build", "run"]


def test_cli_stops_nano_before_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: calls.append(list(services)) or True,
    )
    monkeypatch.setattr(_model_servers, "_build_processes", lambda: [])
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: calls.append("run"))
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    _model_servers.run()

    assert calls == [[("agent-llm", 8107)], "run"]


def test_cli_aborts_when_replaced_model_cannot_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "stop_persistent_servers", lambda _services: False)
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: started.append(True))
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    with pytest.raises(RuntimeError, match="could not stop replaced"):
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
