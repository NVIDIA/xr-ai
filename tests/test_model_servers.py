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
    / "ai-services/llm/nemotron_omni/nemotron_omni_llm_server/__main__.py"
)
_OMNI_SPEC = importlib.util.spec_from_file_location("nemotron_omni_main", _OMNI_PATH)
assert _OMNI_SPEC and _OMNI_SPEC.loader
_omni = importlib.util.module_from_spec(_OMNI_SPEC)
_OMNI_SPEC.loader.exec_module(_omni)


def test_default_stack_uses_nano_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "spark")

    processes = _model_servers._build_processes()

    assert [process.name for process in processes] == ["stt", "agent-llm", "vlm", "embedding"]
    assert [process.port for process in processes] == [8103, 8107, 8100, 8109]


def test_omni_stack_replaces_nano_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes = _model_servers._build_processes("omni")

    assert [process.name for process in processes] == ["stt", "omni", "embedding"]
    assert [process.port for process in processes] == [8103, 8108, 8109]
    assert str(processes[1].config) == "yaml/dual_48G_ada/nemotron_omni_llm_server.yaml"


@pytest.mark.parametrize(
    ("stack", "config_name", "gpu"),
    [
        ("vlm-llm", "embedding_server.yaml", "0"),
        ("omni", "embedding_server_omni.yaml", "1"),
    ],
)
def test_dual_ada_embedding_follows_stack_gpu_layout(
    monkeypatch: pytest.MonkeyPatch,
    stack: str,
    config_name: str,
    gpu: str,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    process = next(p for p in _model_servers._build_processes(stack) if p.name == "embedding")
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


@pytest.mark.parametrize(
    ("stack", "expected"),
    [
        ("omni", [("agent-llm", 8107), ("vlm", 8100)]),
        ("vlm-llm", [("omni", 8108)]),
    ],
)
def test_starting_stack_stops_incompatible_persistent_models(
    monkeypatch: pytest.MonkeyPatch,
    stack: str,
    expected: list[tuple[str, int]],
) -> None:
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: stopped.extend(services) or True,
    )

    _model_servers._stop_incompatible_stack(stack)

    assert stopped == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "vlm-llm"), (["--vlm-llm-stack"], "vlm-llm"), (["--omni-stack"], "omni")],
)
def test_cli_selects_requested_stack(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: str,
) -> None:
    selected: list[str] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_incompatible_stack", lambda _stack: None)
    monkeypatch.setattr(_model_servers, "_build_processes", lambda stack: selected.append(stack) or [])
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["model_servers", *argv])

    _model_servers.run()

    assert selected == [expected]


@pytest.mark.parametrize(
    ("argv", "expected_stopped"),
    [
        (["--omni-stack"], [("agent-llm", 8107), ("vlm", 8100)]),
        (["--vlm-llm-stack"], [("omni", 8108)]),
    ],
)
def test_cli_stops_incompatible_stack_before_starting(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_stopped: list[tuple[str, int]],
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: calls.append(list(services)) or True,
    )
    monkeypatch.setattr(_model_servers, "_build_processes", lambda _stack: [])
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: calls.append("run"))
    monkeypatch.setattr(sys, "argv", ["model_servers", *argv])

    _model_servers.run()

    assert calls == [expected_stopped, "run"]


def test_cli_aborts_when_incompatible_stack_cannot_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "stop_persistent_servers", lambda _services: False)
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: started.append(True))
    monkeypatch.setattr(sys, "argv", ["model_servers", "--omni-stack"])

    with pytest.raises(RuntimeError, match="could not stop incompatible"):
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


def test_stop_needs_no_stack_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_models", lambda: calls.append("stop"))
    monkeypatch.setattr(sys, "argv", ["model_servers", "--stop"])

    _model_servers.run()

    assert calls == ["stop"]
