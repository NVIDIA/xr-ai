# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample model-server composition, scoped lifecycle, and service runtime tests."""
from __future__ import annotations

import importlib.util
import os.path
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from _helpers_model_servers import profile_config
from xr_ai_model_server_stack import _stack, run_model_stack
from xr_ai_models import load_models_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_stack_logging(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "orchestrator.log"
    log_file.parent.mkdir()
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_kw: log_file)

_OMNI_PATH = (
    _REPO_ROOT
    / "services/nemotron-omni-llm/nemotron_omni_llm_server/__main__.py"
)
_OMNI_SPEC = importlib.util.spec_from_file_location("nemotron_omni_main", _OMNI_PATH)
assert _OMNI_SPEC and _OMNI_SPEC.loader
_omni = importlib.util.module_from_spec(_OMNI_SPEC)
_OMNI_SPEC.loader.exec_module(_omni)

_NANO_PATH = (
    _REPO_ROOT
    / "services/nemotron3-nano-llm/nemotron3_nano_llm_server/__main__.py"
)
_NANO_SPEC = importlib.util.spec_from_file_location("nemotron3_nano_main", _NANO_PATH)
assert _NANO_SPEC and _NANO_SPEC.loader
_nano = importlib.util.module_from_spec(_NANO_SPEC)
_NANO_SPEC.loader.exec_module(_nano)

_EMBEDDING_PATH = (
    _REPO_ROOT / "services/embedding-server/embedding_server/__main__.py"
)
_EMBEDDING_SPEC = importlib.util.spec_from_file_location(
    "embedding_server_main", _EMBEDDING_PATH
)
assert _EMBEDDING_SPEC and _EMBEDDING_SPEC.loader
_embedding = importlib.util.module_from_spec(_EMBEDDING_SPEC)
_EMBEDDING_SPEC.loader.exec_module(_embedding)


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


@pytest.mark.parametrize(
    ("compute_major", "spark_uma", "expected_seqs"),
    [
        (8, False, "384"),
        (12, False, "384"),
        (12, True, "4"),
    ],
)
def test_omni_applies_spark_sequence_default(
    monkeypatch: pytest.MonkeyPatch,
    compute_major: int,
    spark_uma: bool,
    expected_seqs: str,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(_omni, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _omni,
        "load_config",
        lambda: ({}, Path("."), None),
    )
    monkeypatch.setattr(_omni, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_omni, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_omni, "gpu_compute_major", lambda: compute_major)
    monkeypatch.setattr(_omni, "gpu_is_dgx_spark", lambda: spark_uma)
    monkeypatch.setattr(_omni, "serve", lambda **kwargs: captured.update(kwargs))

    _omni.run()

    args = captured["extra_serve_args"]
    assert args[args.index("--max-num-seqs") + 1] == expected_seqs
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert "--kv-cache-memory-bytes" not in args
    assert captured["extra_pip"] == []


@pytest.mark.parametrize(
    ("compute_major", "spark_uma", "expected_seqs"),
    [
        (8, False, "8"),
        (12, True, "4"),
    ],
)
def test_nano_standalone_applies_spark_sequence_default(
    monkeypatch: pytest.MonkeyPatch,
    compute_major: int,
    spark_uma: bool,
    expected_seqs: str,
) -> None:
    captured: dict[str, object] = {}
    config_path = (
        _REPO_ROOT
        / "services/nemotron3-nano-llm/nemotron3_nano_llm_server.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    monkeypatch.setattr(_nano, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _nano, "load_config", lambda: (config, config_path.parent, None)
    )
    monkeypatch.setattr(_nano, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_nano, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_nano, "gpu_compute_major", lambda: compute_major)
    monkeypatch.setattr(_nano, "gpu_is_dgx_spark", lambda: spark_uma)
    monkeypatch.setattr(
        _nano,
        "_ensure_reasoning_parser",
        lambda *_a, **_k: Path("parser.py"),
    )
    monkeypatch.setattr(_nano, "serve", lambda **kwargs: captured.update(kwargs))

    _nano.run()

    args = captured["extra_serve_args"]
    assert args[args.index("--max-num-seqs") + 1] == expected_seqs


@pytest.mark.parametrize(
    ("profile", "compute_major", "expected_utilization", "expected_spark_uma"),
    [
        ("96G_blackwell", 10, "0.35", False),
        ("dual_48G_ada", 8, "0.78", False),
        ("spark", 12, "0.25", True),
    ],
)
def test_omni_profiles_forward_admission_and_explicit_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    compute_major: int,
    expected_utilization: str,
    expected_spark_uma: bool,
) -> None:
    captured: dict[str, object] = {}
    config_path = profile_config("nemotron_omni_llm_server", profile)
    config = yaml.safe_load(config_path.read_text())
    monkeypatch.setattr(_omni, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _omni,
        "load_config",
        lambda: (config, config_path.parent, None),
    )
    monkeypatch.setattr(_omni, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_omni, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_omni, "gpu_compute_major", lambda: compute_major)
    monkeypatch.setattr(_omni, "gpu_is_dgx_spark", lambda: False)
    monkeypatch.setattr(_omni, "serve", lambda **kwargs: captured.update(kwargs))

    _omni.run()

    args = captured["extra_serve_args"]
    cache_index = args.index("--kv-cache-memory-bytes")
    assert args[cache_index + 1] == "2147483648"
    memory_index = args.index("--gpu-memory-utilization")
    assert args[memory_index + 1] == expected_utilization
    assert captured["spark_uma"] is expected_spark_uma


@pytest.mark.parametrize("value", [True, 0, -1, "invalid"])
def test_omni_rejects_invalid_explicit_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    monkeypatch.setattr(_omni, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _omni,
        "load_config",
        lambda: ({"kv_cache_memory_bytes": value}, Path("."), None),
    )
    monkeypatch.setattr(_omni, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_omni, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_omni, "gpu_compute_major", lambda: 12)

    with pytest.raises(SystemExit, match="1"):
        _omni.run()


def test_embedding_default_cache_tracks_service_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[Path] = []
    captured: dict[str, object] = {}
    project = _REPO_ROOT / "services/embedding-server"

    def resolve_cache(_cfg: dict, yaml_dir: Path, *, default: str) -> Path:
        path = Path(os.path.normpath(yaml_dir / default))
        resolved.append(path)
        return path

    monkeypatch.setattr(_embedding, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "load_config", lambda: ({}, project, None))
    monkeypatch.setattr(_embedding, "resolve_model_cache", resolve_cache)
    monkeypatch.setattr(_embedding, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "serve", lambda **kwargs: captured.update(kwargs))

    _embedding.run()

    assert resolved == [_REPO_ROOT / "models"]
    assert captured["model"] == "nvidia/Nemotron-3-Embed-1B-BF16"


@pytest.mark.parametrize("backend", ["pip", "docker"])
@pytest.mark.parametrize("attention_backend", [None, "TRITON_ATTN", "FLASH_ATTN"])
def test_embedding_only_forwards_configured_attention_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, attention_backend: str | None,
) -> None:
    captured: dict[str, object] = {}
    config = {"vllm_backend": backend}
    if attention_backend is not None:
        config["attention_backend"] = attention_backend
    monkeypatch.setattr(_embedding, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "load_config", lambda: (config, Path("."), None))
    monkeypatch.setattr(_embedding, "resolve_model_cache", lambda *_a, **_k: Path("models"))
    monkeypatch.setattr(_embedding, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "serve", lambda **kwargs: captured.update(kwargs))

    _embedding.run()

    assert captured["backend"] == backend
    args = captured["extra_serve_args"]
    if attention_backend is None:
        assert "--attention-backend" not in args
    else:
        assert args[args.index("--attention-backend") + 1] == attention_backend


@pytest.mark.parametrize("profile", ["spark", "96G_blackwell", "dual_48G_ada", "standalone"])
def test_embedding_profiles_launch_nemotron3_with_existing_endpoint(
    monkeypatch: pytest.MonkeyPatch, profile: str,
) -> None:
    captured: dict[str, object] = {}
    config_path = (
        _REPO_ROOT / "services/embedding-server/embedding_server.yaml"
        if profile == "standalone" else profile_config("embedding_server", profile)
    )
    config = yaml.safe_load(config_path.read_text())
    monkeypatch.setattr(_embedding, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _embedding,
        "load_config",
        lambda: (config, config_path.parent, None),
    )
    monkeypatch.setattr(
        _embedding,
        "resolve_model_cache",
        lambda *_a, **_k: Path("models"),
    )
    monkeypatch.setattr(_embedding, "setup_hf_env", lambda *_a, **_k: None)
    monkeypatch.setattr(_embedding, "serve", lambda **kwargs: captured.update(kwargs))

    _embedding.run()

    assert captured["model"] == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert captured["port"] == 8109
    assert captured["spark_uma"] is (profile == "spark")
    args = captured["extra_serve_args"]
    assert args[args.index("--served-model-name") + 1] == "embed"
    if profile == "spark":
        assert args[args.index("--attention-backend") + 1] == "TRITON_ATTN"
    else:
        assert "--attention-backend" not in args


def test_embedding_profiles_retain_fractional_memory_budget() -> None:
    for profile in ("spark", "96G_blackwell", "dual_48G_ada"):
        config_path = profile_config("embedding_server", profile)
        config = yaml.safe_load(config_path.read_text())
        assert config["gpu_memory_utilization"] == 0.08
        assert "kv_cache_memory_bytes" not in config




_SAMPLES = {
    "procedure-capture-guide": ("stt", "tts", "omni", "vlm"),
    "simple-vlm-example": ("speaker-stt", "tts", "vlm"),
    "lab-instrument-monitoring": ("speaker-stt", "tts", "omni", "vlm"),
    "tea-making-sample": ("speaker-stt", "tts", "omni", "embedding"),
    "xr-render-demo": ("speaker-stt", "tts", "omni", "vlm"),
}


@pytest.mark.parametrize("sample,services", _SAMPLES.items())
@pytest.mark.parametrize("profile", ["spark", "dual_48G_ada", "96G_blackwell"])
def test_sample_launch_and_stop_use_only_required_servers(monkeypatch, tmp_path, sample, services, profile):
    base = _REPO_ROOT / "agent-samples" / sample / "model-servers"
    spec = importlib.util.spec_from_file_location("sample_models", base / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._SERVICES == services
    configs = _stack._configs(base, module._SERVICES, profile)
    processes = _stack._processes(configs, tmp_path, _REPO_ROOT)
    assert tuple(process.name for process in processes) == services
    model_file = "models.local.json" if sample == "tea-making-sample" else "models.json"
    models = load_models_config(base.parent / "yaml" / model_file)
    ports = {config["port"] for config in configs.values()}
    for spec in models.entries.values():
        endpoint = urlsplit(spec.endpoint.base_url)
        assert endpoint.hostname in {"localhost", "127.0.0.1"}
        assert endpoint.port in ports
    for process in processes:
        assert (Path(process.project) / "pyproject.toml").is_file()
        assert process.launch_mode == "persist"
        config = yaml.safe_load(Path(process.config).read_text())
        assert config["model_cache"] == str((_REPO_ROOT / "models").resolve())
        assert process.port == config["port"]
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: pytest.fail("stop must not require credentials"))
    monkeypatch.setattr(_stack, "run_stack", lambda *_a, **_k: pytest.fail("stop must not launch"))
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: pytest.fail("explicit profile must bypass detection"))
    run_model_stack(base, services, stop=True, gpu_profile=profile)
    assert stopped == [(name, configs[name]["port"]) for name in services]


def test_start_materializes_configs_through_readiness_and_does_not_stop(monkeypatch):
    base = _REPO_ROOT / "agent-samples/simple-vlm-example/model-servers"
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: "spark")
    credentials = []
    monkeypatch.setattr(_stack, "require_credentials", lambda name: credentials.append(name))
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda *_a: pytest.fail("start must not sweep servers"))
    launched = []
    def launch(processes, root, *, exit_after_ready):
        assert root == base
        assert exit_after_ready is True
        assert [p.name for p in processes] == ["speaker-stt", "tts", "vlm"]
        for process in processes:
            assert Path(process.config).is_file()
            launched.append(Path(process.config))
    monkeypatch.setattr(_stack, "run_stack", launch)
    run_model_stack(base, _SAMPLES["simple-vlm-example"])
    assert credentials == ["HF_TOKEN"]
    assert launched and not any(path.exists() for path in launched)


def test_stop_failure_is_reported(monkeypatch):
    base = _REPO_ROOT / "agent-samples/simple-vlm-example/model-servers"
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda *_a: False)
    with pytest.raises(RuntimeError, match="could not be stopped"):
        run_model_stack(base, _SAMPLES["simple-vlm-example"], stop=True, gpu_profile="spark")


@pytest.mark.parametrize("sample, services", _SAMPLES.items())
def test_default_stop_needs_no_gpu_or_credentials(monkeypatch, sample, services):
    from xr_ai_launcher import GPUInventoryError

    base = _REPO_ROOT / "agent-samples" / sample / "model-servers"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    def failed_inventory():
        raise GPUInventoryError("nvidia-smi unavailable")
    monkeypatch.setattr(_stack, "detect_gpu_config", failed_inventory)
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: pytest.fail("stop required credentials"))
    monkeypatch.setattr(_stack, "run_stack", lambda *_a, **_kw: pytest.fail("stop launched servers"))
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    run_model_stack(base, services, stop=True)
    expected_ports = {"stt": 8103, "speaker-stt": 8102, "tts": 8105, "omni": 8108, "vlm": 8100, "embedding": 8109}
    assert stopped == [(service, expected_ports[service]) for service in services]


def test_failed_readiness_retains_private_merged_configs_without_credentials(monkeypatch, capsys):
    from xr_ai_launcher._failure import FailureContext, emit_failure_summary

    base = _REPO_ROOT / "agent-samples/simple-vlm-example/model-servers"
    original_configs = _stack._configs
    def configured(*args):
        configs = original_configs(*args)
        configs["vlm"]["hf_token"] = "inline-credential"
        return configs
    monkeypatch.setattr(_stack, "_configs", configured)
    monkeypatch.setenv("HF_TOKEN", "environment-credential")
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: None)
    captured = []
    def fail(processes, *_a, **_kw):
        captured.extend(Path(process.config) for process in processes)
        vlm = processes[-1]
        path = Path(vlm.config)
        assert yaml.safe_load(path.read_text())["hf_token"] == "inline-credential"
        emit_failure_summary(FailureContext(vlm.name, ("server", "--config", str(path)),
                                           Path(vlm.project), path), "readiness", returncode=1)
        raise SystemExit(1)
    monkeypatch.setattr(_stack, "run_stack", fail)
    with pytest.raises(SystemExit, match="1"):
        run_model_stack(base, _SAMPLES["simple-vlm-example"], gpu_profile="spark")
    assert captured
    for path in captured:
        assert path.is_file()
        assert path.parent.parent.name == "logs"
        assert path.parent.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
        text = path.read_text()
        assert "credential" not in text
        config = yaml.safe_load(text)
        assert config["port"] in {8100, 8102, 8105}
    assert str(captured[-1]) in capsys.readouterr().err


def test_override_paths_and_ports_are_used_for_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: pytest.fail("explicit stop detected GPU"))
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: pytest.fail("stop required credentials"))
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text('stt:\n  port: 9213\n  model_cache: ../cache\n')
    config = _stack._configs(tmp_path, ("stt",), "custom")["stt"]
    assert config["model_cache"] == str(tmp_path / "cache")
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    run_model_stack(tmp_path, ("stt",), stop=True, gpu_profile="custom")
    assert stopped == [("stt", 9213)]


def test_clef_is_selected_with_service_defaults(tmp_path):
    base = tmp_path / "model-servers"
    (base / "yaml").mkdir(parents=True)
    (base / "yaml/custom.yaml").write_text("clef: {device: cuda:1, port: 9210}\n")
    processes = _stack._processes(_stack._configs(base, ("clef",), "custom"), tmp_path, _REPO_ROOT)
    assert [(p.name, p.command, p.port, p.launch_mode) for p in processes] == [
        ("clef", "clef_server", 9210, "persist"),
    ]
    assert Path(processes[0].project) == _REPO_ROOT / "services/clef-server"
    config = yaml.safe_load(Path(processes[0].config).read_text())
    assert config["model_cache"] == str((_REPO_ROOT / "models").resolve())
    assert config["device"] == "cuda:1"


@pytest.mark.parametrize("profile,port", [(None, 8120), ("custom", 9210)])
def test_clef_stop_uses_selected_port(monkeypatch, tmp_path, profile, port):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text("clef: {port: 9210}\n")
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: pytest.fail("stop detected GPU"))
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: pytest.fail("stop required credentials"))
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    run_model_stack(tmp_path, ("clef",), stop=True, gpu_profile=profile)
    assert stopped == [("clef", port)]


@pytest.mark.parametrize("body", [
    "{}", "stt: []", "stt: {port: 0}", "stt: {port: true}",
    "stt: {port: 65536}", "stt: [", "stt: {model_cache: null}",
])
def test_invalid_profile_fails_before_any_lifecycle_action(tmp_path, body):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text(body)
    with pytest.raises(ValueError):
        _stack._configs(tmp_path, ("stt",), "custom")


def test_unknown_profile_reports_choices():
    base = _REPO_ROOT / "agent-samples/simple-vlm-example/model-servers"
    with pytest.raises(ValueError, match="unknown GPU profile.*available"):
        _stack._configs(base, _SAMPLES["simple-vlm-example"], "../spark")


@pytest.mark.parametrize("sample", _SAMPLES)
@pytest.mark.parametrize("argv,code", [(["--help"], 0), (["--models", "x"], 2), (["--allow-anonymous"], 2)])
def test_cli_help_and_removed_options_do_not_touch_servers(monkeypatch, sample, argv, code):
    spec = importlib.util.spec_from_file_location(
        "sample_models", _REPO_ROOT / "agent-samples" / sample / "model-servers/main.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_model_stack", lambda *_a, **_k: pytest.fail("must not touch services"))
    monkeypatch.setattr(sys, "argv", ["models", *argv])
    with pytest.raises(SystemExit) as error:
        module.run()
    assert error.value.code == code


def _repository_yaml_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.yaml"):
        relative_parts = path.relative_to(root).parts
        if relative_parts[:1] == ("apps",) or ".venv" in relative_parts:
            continue
        paths.append(path)
    return sorted(paths)


def test_all_shipped_vllm_images_track_dispatcher_default() -> None:
    configured_images: dict[Path, str] = {}
    for path in _repository_yaml_paths(_REPO_ROOT):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and "vllm_image" in config:
            configured_images[path] = config["vllm_image"]

    assert configured_images
    assert set(configured_images.values()) == {_nano.DEFAULT_IMAGE}


def test_repository_yaml_paths_exclude_apps_and_virtual_environments(
    tmp_path: Path,
) -> None:
    included = tmp_path / "services" / "server.yaml"
    nested_apps = tmp_path / "services" / "apps" / "helper.yaml"
    app = tmp_path / "apps" / "private-app" / "config.yaml"
    virtualenv = tmp_path / "services" / "worker" / ".venv" / "config.yaml"
    for path in (included, nested_apps, app, virtualenv):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enabled: true\n")

    assert _repository_yaml_paths(tmp_path) == sorted((included, nested_apps))
