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
from xr_ai_vllm import _docker as _vllm_docker

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PATH = _REPO_ROOT / "agent-samples/model-servers/main.py"
_SPEC = importlib.util.spec_from_file_location("model_servers_main", _MAIN_PATH)
assert _SPEC and _SPEC.loader
_model_servers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_model_servers)

_SIMPLE_MAIN_PATH = _REPO_ROOT / "agent-samples/simple-vlm-example/main.py"
_SIMPLE_SPEC = importlib.util.spec_from_file_location(
    "simple_vlm_example_main", _SIMPLE_MAIN_PATH
)
assert _SIMPLE_SPEC and _SIMPLE_SPEC.loader
_simple_vlm = importlib.util.module_from_spec(_SIMPLE_SPEC)
_SIMPLE_SPEC.loader.exec_module(_simple_vlm)

_VLM_MAIN_PATH = _REPO_ROOT / "services/vlm-server/vlm_server/__main__.py"
_VLM_SPEC = importlib.util.spec_from_file_location(
    "model_server_fingerprint_vlm", _VLM_MAIN_PATH
)
assert _VLM_SPEC and _VLM_SPEC.loader
_vlm_server = importlib.util.module_from_spec(_VLM_SPEC)
_VLM_SPEC.loader.exec_module(_vlm_server)

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


def _vlm_launch_fingerprint(
    config_path: Path,
    hf_token: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> str | None:
    config = yaml.safe_load(config_path.read_text())
    raw_cache = Path(config["model_cache"])
    model_cache = (
        raw_cache
        if raw_cache.is_absolute()
        else (config_path.parent / raw_cache).resolve()
    )
    captured: dict = {}

    monkeypatch.setattr(_vlm_server, "setup_logging", lambda _name: None)
    monkeypatch.setattr(
        _vlm_server,
        "load_config",
        lambda: (config, config_path.parent, None),
    )
    monkeypatch.setattr(
        _vlm_server,
        "resolve_model_cache",
        lambda *_args, **_kwargs: model_cache,
    )
    monkeypatch.setattr(
        _vlm_server,
        "setup_hf_env",
        lambda cfg, _cache: (
            str(cfg["cuda_visible_devices"])
            if "cuda_visible_devices" in cfg
            else None
        ),
    )
    if hf_token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", hf_token)
    monkeypatch.setattr(
        _vllm_docker,
        "run",
        lambda **kwargs: captured.update(kwargs),
    )

    _vlm_server.run()

    argv = _vllm_docker.build_run_argv(
        image=captured["image"],
        container_name=captured["container_name"],
        port=captured["port"],
        model_cache=captured["model_cache"],
        hf_token=captured["hf_token"],
        cuda_visible_devices=captured["cuda_visible_devices"],
        extra_env=captured["extra_env"],
        extra_pip=captured["extra_pip"],
        vllm_argv=captured["vllm_argv"],
    )
    return _vllm_docker._requested_fingerprint(argv)


def test_default_profile_uses_omni_and_cosmos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "spark")

    processes, credentials = _model_servers._build_processes("default")

    assert [process.name for process in processes] == ["stt", "omni", "vlm", "embedding"]
    assert [process.port for process in processes] == [8103, 8108, 8100, 8109]
    assert credentials == ()


@pytest.mark.parametrize("gpu_profile", ["dual_48G_ada", "spark", "96G_blackwell"])
def test_simple_vlm_shares_supported_model_server_vlm_config(
    monkeypatch: pytest.MonkeyPatch,
    gpu_profile: str,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: gpu_profile)
    monkeypatch.setattr(_simple_vlm, "detect_gpu_config", lambda: gpu_profile)

    shared = {
        process.name: process
        for process in _model_servers._build_processes("default")[0]
    }
    sample = {
        process.name: process
        for process in _simple_vlm._build_processes()[0]
    }

    model_servers_config = (
        _REPO_ROOT
        / "agent-samples/model-servers"
        / shared["vlm"].config
    ).resolve()
    simple_vlm_config = (
        _REPO_ROOT
        / "agent-samples/simple-vlm-example"
        / sample["vlm"].config
    ).resolve()
    assert simple_vlm_config == model_servers_config
    assert sample["vlm"].project == shared["vlm"].project
    assert sample["vlm"].command == shared["vlm"].command
    assert sample["vlm"].launch_mode == "own"

    stt_config = (
        _REPO_ROOT
        / "agent-samples/simple-vlm-example"
        / sample["stt"].config
    ).resolve()
    assert stt_config == (
        _REPO_ROOT / "agent-samples/simple-vlm-example/yaml/stt_server.yaml"
    )
    assert "cuda_visible_devices" not in yaml.safe_load(stt_config.read_text())


@pytest.mark.parametrize("gpu_profile", ["dual_48G_ada", "spark", "96G_blackwell"])
@pytest.mark.parametrize("hf_token", [None, "hf_test_token"])
def test_simple_vlm_matches_full_model_server_launch_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    gpu_profile: str,
    hf_token: str | None,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: gpu_profile)
    monkeypatch.setattr(_simple_vlm, "detect_gpu_config", lambda: gpu_profile)

    shared = {
        process.name: process
        for process in _model_servers._build_processes("default")[0]
    }
    sample = {
        process.name: process
        for process in _simple_vlm._build_processes()[0]
    }

    shared_fingerprint = _vlm_launch_fingerprint(
        Path(shared["vlm"].config), hf_token, monkeypatch
    )
    sample_fingerprint = _vlm_launch_fingerprint(
        Path(sample["vlm"].config), hf_token, monkeypatch
    )

    assert shared_fingerprint is not None
    assert sample_fingerprint == shared_fingerprint


def test_simple_vlm_uses_standalone_config_on_smaller_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_hardware() -> str:
        raise _simple_vlm.GPUInventoryError("unsupported GPU topology")

    monkeypatch.setattr(_simple_vlm, "detect_gpu_config", unsupported_hardware)

    sample = {
        process.name: process
        for process in _simple_vlm._build_processes()[0]
    }
    vlm_config = (
        _REPO_ROOT
        / "agent-samples/simple-vlm-example"
        / sample["vlm"].config
    ).resolve()
    config = yaml.safe_load(vlm_config.read_text())

    assert vlm_config == (
        _REPO_ROOT / "agent-samples/simple-vlm-example/yaml/vlm_server.yaml"
    )
    assert config["gpu_memory_utilization"] == 0.85
    assert "cuda_visible_devices" not in config


@pytest.mark.parametrize(
    "profile_name",
    ["models.local.json", "models.hosted.json", "models.omni.json"],
)
def test_simple_vlm_profiles_keep_stt_standalone(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
) -> None:
    profile_path = (
        _REPO_ROOT / "agent-samples/simple-vlm-example/yaml" / profile_name
    )
    monkeypatch.setattr(
        _simple_vlm,
        "load_model_deployment",
        lambda _worker: _model_servers.load_deployment_profile(profile_path),
    )
    def unsupported_hardware() -> str:
        raise _simple_vlm.GPUInventoryError("unsupported GPU topology")

    monkeypatch.setattr(_simple_vlm, "detect_gpu_config", unsupported_hardware)

    sample = {
        process.name: process
        for process in _simple_vlm._build_processes()[0]
    }
    stt_config = (
        _REPO_ROOT
        / "agent-samples/simple-vlm-example"
        / sample["stt"].config
    ).resolve()

    assert stt_config == (
        _REPO_ROOT / "agent-samples/simple-vlm-example/yaml/stt_server.yaml"
    )


def test_simple_vlm_rejects_missing_shared_vlm_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(_simple_vlm, "detect_gpu_config", lambda: "spark")
    monkeypatch.setattr(_simple_vlm, "_MODEL_SERVERS_YAML", tmp_path)

    with pytest.raises(FileNotFoundError, match="shared VLM config does not exist"):
        _simple_vlm._build_processes()

def test_explicit_gpu_profile_bypasses_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_detection() -> str:
        raise AssertionError("automatic detection must not run for an explicit profile")

    monkeypatch.setattr(_model_servers, "detect_gpu_config", fail_detection)

    processes, _ = _model_servers._build_processes("default", "spark")

    expected_dir = _REPO_ROOT / "agent-samples/model-servers/yaml/spark"
    assert all(Path(process.config).parent == expected_dir for process in processes)


def test_nim_profile_mixes_nim_containers_and_local_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_model_servers, "detect_gpu_config", lambda: "dual_48G_ada")

    processes, credentials = _model_servers._build_processes("vlm_llm_nim")

    assert [process.name for process in processes] == [
        "llm-nim", "vlm-nim", "stt", "embedding",
    ]
    assert [process.port for process in processes] == [8110, 8100, 8103, 8109]
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
        ("default", "embedding", "embedding_server.yaml", "0"),
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
        ("llm-nim", 8110),
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
