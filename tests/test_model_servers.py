# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared model-server stack."""
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


def test_stack_uses_omni_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "spark")

    processes = _model_servers._build_processes()

    assert [process.name for process in processes] == ["stt", "omni", "vlm", "embedding"]
    assert [process.port for process in processes] == [8103, 8108, 8100, 8109]
    assert str(processes[1].config) == "yaml/spark/nemotron_omni_llm_server.yaml"


def test_dual_ada_places_omni_and_cosmos_on_separate_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    configs = {
        process.name: yaml.safe_load(
            (_REPO_ROOT / "agent-samples/model-servers" / str(process.config)).read_text()
        )
        for process in _model_servers._build_processes()
    }

    assert {configs[name]["cuda_visible_devices"] for name in ("vlm", "embedding")} == {"0"}
    assert {configs[name]["cuda_visible_devices"] for name in ("omni", "stt")} == {"1"}


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


def test_starting_stack_stops_legacy_nano(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda services: stopped.extend(services) or True,
    )

    _model_servers._stop_legacy_models()

    assert stopped == [("agent-llm", 8107)]


def test_cli_starts_the_shared_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[bool] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "warn_if_missing", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_legacy_models", lambda: None)
    monkeypatch.setattr(_model_servers, "_build_processes", lambda: selected.append(True) or [])
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    _model_servers.run()

    assert selected == [True]


@pytest.mark.parametrize("option", ["--omni-stack", "--vlm-llm-stack"])
def test_removed_stack_selectors_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["model_servers", option])

    with pytest.raises(SystemExit):
        _model_servers.run()


def test_cli_stops_legacy_nano_before_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "warn_if_missing", lambda *_a, **_k: None)
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


def test_cli_aborts_when_legacy_nano_cannot_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "warn_if_missing", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "stop_persistent_servers", lambda _services: False)
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: started.append(True))
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    with pytest.raises(RuntimeError, match="could not stop legacy"):
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
