# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared, profile-driven model-server stack."""
from __future__ import annotations

import importlib.util
import json
import os.path
import shlex
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


def test_default_profile_uses_omni_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "spark")

    processes, credentials = _model_servers._build_processes("default")

    assert [process.name for process in processes] == [
        "stt", "tts", "omni", "vlm", "embedding",
    ]
    assert [process.port for process in processes] == [8103, 8105, 8108, 8100, 8109]
    tts = next(process for process in processes if process.name == "tts")
    assert tts.project == "../../services/piper-tts"
    assert tts.command == "piper_tts_server"
    assert Path(tts.config).name == "piper_tts_server.yaml"
    assert tts.launch_mode == "persist"
    assert credentials == ()


def test_explicit_gpu_profile_bypasses_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_detection() -> str:
        raise AssertionError("automatic detection must not run for an explicit profile")

    monkeypatch.setattr(_model_servers, "detect_gpu_config", fail_detection)

    processes, _ = _model_servers._build_processes("default", "spark")

    expected_dir = _REPO_ROOT / "agent-samples/model-servers/yaml/spark"
    assert all(Path(process.config).parent == expected_dir for process in processes)


def test_service_catalog_does_not_duplicate_yaml_ports() -> None:
    assert all(len(service) == 3 for service in _model_servers._MODEL_SERVICES.values())


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("port: 8100\n", 8100),
        ("http_port: 9010\n", 9010),
        ("port: '8109' # API\n", 8109),
        ("host: 0.0.0.0\n", None),
    ],
)
def test_read_service_port_uses_top_level_yaml(
    tmp_path: Path, body: str, expected: int | None,
) -> None:
    config = tmp_path / "service.yaml"
    config.write_text(body, encoding="utf-8")

    assert _model_servers._read_service_port(config) == expected


@pytest.mark.parametrize("value", ["not-a-port", "0", "65536"])
def test_read_service_port_rejects_invalid_values(
    tmp_path: Path, value: str,
) -> None:
    config = tmp_path / "service.yaml"
    config.write_text(f"port: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="port"):
        _model_servers._read_service_port(config)


def test_known_ports_are_discovered_from_service_yaml() -> None:
    assert set(_model_servers._known_service_ports()) == {
        ("stt-nim", 9010),
        ("tts-nim", 9011),
        ("llm-nim", 8110),
        ("vlm-nim", 8100),
        ("stt", 8103),
        ("tts", 8105),
        ("agent-llm", 8107),
        ("omni", 8108),
        ("vlm", 8100),
        ("embedding", 8109),
    }


def test_nim_profile_mixes_nim_containers_and_local_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, credentials = _model_servers._build_processes("vlm_llm_nim")

    assert [process.name for process in processes] == [
        "llm-nim", "vlm-nim", "stt", "tts", "embedding",
    ]
    assert [process.port for process in processes] == [8110, 8100, 8103, 8105, 8109]
    assert credentials == ("NGC_API_KEY",)


@pytest.mark.parametrize(
    "config_path",
    sorted(
        (_REPO_ROOT / "agent-samples/model-servers/yaml").glob(
            "*/nim_vlm_server.yaml"
        )
    ),
)
def test_nim_profiles_serve_cosmos3_nano_reasoner(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())

    assert config["image"] == "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0"
    assert config["env"]["NIM_MODEL_SIZE"] == "nano"


@pytest.mark.parametrize(
    "config_path",
    sorted(
        (_REPO_ROOT / "agent-samples/model-servers/yaml").glob(
            "*/nim_llm_server.yaml"
        )
    ),
)
def test_nim_profiles_serve_nemotron_omni(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    env = config["env"]
    args = shlex.split(env["NIM_PASSTHROUGH_ARGS"])
    expected_budget = {
        "spark": "0.35",
        "96G_blackwell": "0.4",
        "dual_48G_ada": "0.8",
    }[config_path.parent.name]

    assert config["image"] == (
        "nvcr.io/nim/nvidia/"
        "nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant"
    )
    assert "NIM_KVCACHE_PERCENT" not in env
    memory_index = args.index("--gpu-memory-utilization")
    assert args[memory_index + 1] == expected_budget
    assert "--reasoning-parser" in args
    assert args[args.index("--reasoning-parser") + 1] == "nemotron_v3"
    assert "--tool-call-parser" in args
    assert args[args.index("--tool-call-parser") + 1] == "qwen3_coder"


def test_custom_profiles_can_still_launch_riva_speech_nims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "models.custom_riva.json"
    profile.write_text(
        json.dumps(
            {
                "models": {
                    role: {
                        "adapter": {"kind": "riva_grpc"},
                        "endpoint": {"base_url": endpoint},
                        "deployment": {
                            "ownership": "managed",
                            "service": service,
                            "credentials": ["NGC_API_KEY"],
                        },
                    }
                    for role, endpoint, service in (
                        ("stt", "localhost:50051", "stt-nim"),
                        ("tts", "localhost:50052", "tts-nim"),
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, credentials = _model_servers._build_processes(str(profile))

    assert [process.name for process in processes] == ["stt-nim", "tts-nim"]
    assert credentials == ("NGC_API_KEY",)


@pytest.mark.parametrize(
    ("selection", "service", "config_name", "gpu"),
    [
        ("default", "embedding", "embedding_server.yaml", "0"),
        ("vlm_llm_nim", "embedding", "embedding_server.yaml", "0"),
        ("vlm_llm_nim", "stt", "stt_server.yaml", "1"),
        ("vlm_llm_nim", "vlm-nim", "nim_vlm_server.yaml", "0"),
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


@pytest.mark.parametrize(
    "profile_path",
    sorted(
        (_REPO_ROOT / "agent-samples/model-servers/yaml").glob(
            "*/nemotron_omni_llm_server*.yaml"
        )
    ),
)
def test_omni_profiles_select_supported_vllm_configuration(profile_path: Path) -> None:
    config = yaml.safe_load(profile_path.read_text())

    assert config["vllm_backend"] == "docker"
    assert config["vllm_image"] == "vllm/vllm-openai:v0.20.0"
    assert config["extra_pip"] == []
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
        ("llm-nim", 8110),
        ("vlm-nim", 8100),
        ("stt", 8103),
        ("tts", 8105),
        ("agent-llm", 8107),
        ("omni", 8108),
        ("vlm", 8100),
        ("embedding", 8109),
    }


def test_stop_fails_when_any_service_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _model_servers,
        "stop_persistent_servers",
        lambda _services: False,
    )

    with pytest.raises(SystemExit, match="one or more persistent servers"):
        _model_servers._stop_models()


@pytest.mark.parametrize(
    ("selection", "expected_stopped_ports"),
    [
        # The selected profile's ports are kept; everything else is stopped.
        ("default", {9010, 9011, 8110, 8107}),
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
        ([], "default"),
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
        lambda selection, _gpu_profile=None: (selected.append(selection) or [], ()),
    )
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", ["model_servers", *argv])

    _model_servers.run()

    assert selected == [expected]


def test_cli_passes_explicit_gpu_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[tuple[str, str | None]] = []
    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "require_credentials", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_stop_unselected_services", lambda _p: None)
    monkeypatch.setattr(
        _model_servers,
        "_build_processes",
        lambda selection, gpu_profile=None: (
            selected.append((selection, gpu_profile)) or [], ()
        ),
    )
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: None)
    monkeypatch.setattr(
        sys, "argv", ["model_servers", "--gpu-profile", "dual_48G_ada"],
    )

    _model_servers.run()

    assert selected == [("default", "dual_48G_ada")]


def test_invalid_gpu_profile_name_is_rejected() -> None:
    with pytest.raises(
        _model_servers.argparse.ArgumentTypeError, match="unknown GPU profile",
    ):
        _model_servers._gpu_profile_name("does-not-exist")


def test_gpu_profile_discovery_tolerates_missing_yaml_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_model_servers, "_BASE", tmp_path)

    assert _model_servers._gpu_profile_names() == ()


def test_custom_gpu_profile_must_contain_every_selected_service_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "yaml" / "custom").mkdir(parents=True)
    profile = _REPO_ROOT / "agent-samples/model-servers/yaml/models.default.json"
    monkeypatch.setattr(_model_servers, "_BASE", tmp_path)

    with pytest.raises(ValueError, match="profile 'custom' is incomplete.*stt_server"):
        _model_servers._build_processes(str(profile), "custom")


def test_selected_service_config_must_declare_http_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "yaml" / "custom" / "stt_server.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("host: 0.0.0.0\n", encoding="utf-8")
    profile = _REPO_ROOT / "agent-samples/model-servers/yaml/models.default.json"
    monkeypatch.setattr(_model_servers, "_BASE", tmp_path)

    with pytest.raises(
        ValueError,
        match=r"stt_server\.yaml: service config must declare port or http_port",
    ):
        _model_servers._build_processes(str(profile), "custom")


def test_cli_reports_gpu_inventory_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_inventory(*_args, **_kwargs):
        raise _model_servers.GPUInventoryError("GPU memory telemetry unavailable")

    monkeypatch.setattr(_model_servers, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_model_servers, "_build_processes", fail_inventory)
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    with pytest.raises(SystemExit) as error:
        _model_servers.run()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "model_servers: error: GPU memory telemetry unavailable" in stderr
    assert "--gpu-profile NAME" in stderr
    assert "Traceback" not in stderr


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
    assert Path(processes[0].config) == (
        _REPO_ROOT
        / "agent-samples/model-servers/yaml/dual_48G_ada/vlm_server.yaml"
    )


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
        lambda _selection, _gpu_profile=None: ([], ("NGC_API_KEY",)),
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
        _model_servers, "_build_processes",
        lambda _selection, _gpu_profile=None: ([], ()),
    )
    monkeypatch.setattr(_model_servers, "run_stack", lambda *_a, **_k: started.append(True))
    monkeypatch.setattr(sys, "argv", ["model_servers"])

    with pytest.raises(RuntimeError, match="could not stop persistent servers"):
        _model_servers.run()

    assert started == []


@pytest.mark.parametrize("moe_backend", ["cutlass", None])
def test_omni_only_forwards_configured_moe_backend(
    monkeypatch: pytest.MonkeyPatch,
    moe_backend: str | None,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(_omni, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _omni,
        "load_config",
        lambda: (
            {"moe_backend": moe_backend} if moe_backend is not None else {},
            Path("."),
            None,
        ),
    )
    monkeypatch.setattr(_omni, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_omni, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_omni, "gpu_compute_major", lambda: 10)
    monkeypatch.setattr(_omni, "serve", lambda **kwargs: captured.update(kwargs))

    _omni.run()

    args = captured["extra_serve_args"]
    if moe_backend is None:
        assert "--moe-backend" not in args
    else:
        assert args[args.index("--moe-backend") + 1] == moe_backend


def test_embedding_default_cache_tracks_service_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[Path] = []
    project = _REPO_ROOT / "services/embedding-server"

    def resolve_cache(_cfg: dict, yaml_dir: Path, *, default: str) -> Path:
        path = Path(os.path.normpath(yaml_dir / default))
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
