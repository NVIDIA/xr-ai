# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import device_io_hub.__main__ as device_io_hub
import nim_riva_server.__main__ as riva_server
import nim_riva_server.repository as riva_repository
import pytest
import stt_server.__main__ as stt
import xr_ai_vllm
from xr_ai_vllm import _docker, _nim, _pip

_ROOT = Path(__file__).resolve().parents[1]


def _load_main(project: str, package: str):
    spec = importlib.util.spec_from_file_location(
        f"{package}_prepare", _ROOT / "services" / project / package / "__main__.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pocket = _load_main("pocket-tts", "pocket_tts_server")


@pytest.fixture(autouse=True)
def _stable_cache_env(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    cache_names = ("HF_HOME", "HF_XET_CACHE", "HF_XET_HIGH_PERFORMANCE", "NEMO_CACHE_DIR")
    for name in cache_names:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("backend", ["pip", "docker"])
def test_vllm_prepare_downloads_with_selected_backend_and_never_serves(
    backend, tmp_path, monkeypatch,
):
    calls: list[str] = []
    commands: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(_pip, "prepare", lambda _model: calls.append("pip"))
    monkeypatch.setattr(_docker, "_maybe_ngc_login", lambda _image: None)
    monkeypatch.setattr(
        _docker.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    monkeypatch.setattr(
        _pip, "run", lambda **_kwargs: pytest.fail("vLLM must not serve"),
    )
    monkeypatch.setattr(
        _docker, "run", lambda **_kwargs: pytest.fail("vLLM must not serve"),
    )
    xr_ai_vllm.serve(
        backend=backend, persistent=True, container_name="model",
        log_prefix="model", model="org/model", extra_serve_args=[],
        host="0.0.0.0", port=8100, model_cache=tmp_path, prepare=True,
    )
    assert calls == (["pip"] if backend == "pip" else [])
    assert bool(commands) == (backend == "docker")
    if commands:
        command, options = commands[0]
        payload = command[command.index("-c") + 1]
        assert payload.endswith("true")
        assert "snapshot_download" in payload
        assert "TQDM_POSITION=-1" in command
        assert command.count("--init") == 1
        assert command.count("--rm") == 1
        assert command.count("--name") == 1
        assert command[command.index("--name") + 1] == (
            f"model-prepare-{os.getpid()}"
        )
        assert options == {"check": True}


def test_nim_prepare_uses_download_only_container_without_ports(
    tmp_path, monkeypatch,
):
    commands: list[list[str]] = []
    monkeypatch.setenv("NGC_API_KEY", "secret")
    monkeypatch.setattr(_docker, "_maybe_ngc_login", lambda _image: None)
    monkeypatch.setattr(
        _docker, "run_container", lambda **_kwargs: pytest.fail("NIM must not serve"),
    )
    monkeypatch.setattr(
        _nim.subprocess, "run",
        lambda command, **_kwargs: commands.append(command),
    )
    _nim.serve_nim(
        image="nvcr.io/nim/model:1", container_name="model", log_prefix="model",
        http_port=8100, grpc_port=50051, nim_cache=tmp_path, prepare=True,
    )
    command = commands[0]
    assert command[:2] == ["docker", "run"] and command.count("--init") == 1 and "--rm" in command
    assert command[command.index("--entrypoint") + 1] == "download-to-cache"
    assert command[-1] == "nvcr.io/nim/model:1"
    assert "-p" not in command and "--name" not in command


def test_native_model_prepare_downloads_without_loading_server(tmp_path, monkeypatch):
    downloads: list[str] = []
    model_cache = tmp_path / "models"
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda *, repo_id: downloads.append(repo_id),
        ),
    )
    stt._prepare(
        {"model": "org/model", "model_cache": str(model_cache)}, tmp_path,
    )
    assert downloads == ["org/model"]
    assert os.environ["HF_HOME"] == str(model_cache / "huggingface")


@pytest.mark.parametrize(
    ("project", "package", "dispatch_name"),
    [
        ("embedding-server", "embedding_server", "serve"),
        ("llama-nemotron-llm", "llama_nemotron_llm_server", "serve"),
        ("nemotron-omni-llm", "nemotron_omni_llm_server", "serve"),
        ("nemotron3-nano-llm", "nemotron3_nano_llm_server", "serve"),
        ("vlm-server", "vlm_server", "serve"),
        ("nim-server", "nim_server", "serve_nim"),
    ],
)
def test_model_wrapper_cli_prepare_forwards_without_serving(
    project, package, dispatch_name, tmp_path, monkeypatch,
):
    module = _load_main(project, package)
    ready_file = tmp_path / "must-not-be-ready"
    config = {
        "model": "org/model",
        "image": "nvcr.io/nim/model:1",
        "http_port": 8100,
        "model_blackwell": "org/model",
        "model_ada": "org/model",
        "model_bf16": "org/model",
        "use_bf16": True,
    }
    monkeypatch.setattr(module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_config", lambda: (config, tmp_path, ready_file))
    if hasattr(module, "resolve_model_cache"):
        monkeypatch.setattr(module, "resolve_model_cache", lambda *_args, **_kwargs: tmp_path)
    if hasattr(module, "setup_hf_env"):
        monkeypatch.setattr(module, "setup_hf_env", lambda *_args, **_kwargs: None)
    if hasattr(module, "gpu_compute_major"):
        monkeypatch.setattr(module, "gpu_compute_major", lambda: 10)
    if hasattr(module, "_gpu_is_dgx_spark"):
        monkeypatch.setattr(module, "_gpu_is_dgx_spark", lambda: False)
    if hasattr(module, "_ensure_reasoning_parser"):
        monkeypatch.setattr(
            module, "_ensure_reasoning_parser", lambda *_args: tmp_path / "parser.py",
        )

    calls = []

    def dispatch(**kwargs):
        if not kwargs["prepare"]:
            pytest.fail("wrapper entered normal serving mode")
        calls.append(kwargs)

    monkeypatch.setattr(module, dispatch_name, dispatch)
    monkeypatch.setattr(sys, "argv", [package, "--prepare"])

    module.run()

    assert len(calls) == 1
    assert calls[0]["ready_file"] == ready_file
    assert not ready_file.exists()


@pytest.mark.parametrize(
    ("module", "config"),
    [
        (stt, {"model": "org/model"}),
        (pocket, {"voice": "speaker"}),
    ],
)
def test_native_wrapper_cli_prepare_dispatches_without_serving(
    module, config, tmp_path, monkeypatch,
):
    config_path = tmp_path / "service.yaml"
    config_path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()))
    calls = []
    monkeypatch.setattr(module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module, "_prepare", lambda cfg, directory: calls.append((cfg, directory)),
    )
    monkeypatch.setattr(
        module.asyncio,
        "run",
        lambda _awaitable: pytest.fail("wrapper entered normal serving mode"),
    )
    monkeypatch.setattr(
        sys, "argv", [module.__name__, "--prepare", "--config", str(config_path)],
    )

    module.run()

    assert calls == [(config, tmp_path)]


def test_pocket_prepare_forces_cpu_and_loads_artifacts(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str] | str] = []
    model_cache = tmp_path / "models"
    def backend(voice, language, device):
        calls.append((voice, language, device))
        return SimpleNamespace(_ensure_loaded=lambda: calls.append("loaded"))
    monkeypatch.setattr(pocket, "_PocketTTSBackend", backend)
    pocket._prepare(
        {"voice": "speaker", "model_cache": str(model_cache)}, tmp_path,
    )
    assert calls == [("speaker", "english", "cpu"), "loaded"]
    assert os.environ["HF_HOME"] == str(model_cache / "pocket" / "huggingface")


@pytest.mark.parametrize(
    ("inspect_returncode", "expected_events"),
    [
        (0, ["video-codecs", "docker:image"]),
        (1, ["video-codecs", "docker:image", "docker:pull"]),
    ],
    ids=["cached", "uncached"],
)
def test_hub_prepare_validates_video_codecs_before_docker(
    inspect_returncode, expected_events, monkeypatch,
):
    events: list[str] = []
    monkeypatch.setattr(
        device_io_hub,
        "require_nvidia_video_codecs",
        lambda: events.append("video-codecs"),
    )

    def docker(command, **_kwargs):
        events.append(f"docker:{command[1]}")
        return SimpleNamespace(
            returncode=inspect_returncode if command[1] == "image" else 0,
        )

    monkeypatch.setattr(device_io_hub.subprocess, "run", docker)
    monkeypatch.setattr(device_io_hub.sys, "argv", ["device_io_hub", "--prepare"])

    device_io_hub.run()

    assert events == expected_events


def test_riva_prepare_builds_repository_and_does_not_exec_server(
    tmp_path, monkeypatch,
):
    args = riva_server._launch_args(
        {"container_name": "riva", "http_port": 9000, "grpc_port": 50051},
        tmp_path, {"Id": "image-id", "Config": {"Entrypoint": ["serve"], "Cmd": []}},
        prepare=True,
    )
    assert "XR_AI_RIVA_PREPARE_ONLY=1" in args
    assert args.count("--init") == 1
    monkeypatch.setenv("XR_AI_RIVA_COMMAND", '["serve"]')
    monkeypatch.setenv("XR_AI_RIVA_CONTRACT", "{}")
    monkeypatch.setenv("XR_AI_RIVA_PREPARE_ONLY", "1")
    monkeypatch.setenv("NIM_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(
        riva_repository.subprocess, "check_output", lambda *_args, **_kwargs: "GPU, 9.0, 580",
    )
    prepared: list[object] = []
    monkeypatch.setattr(
        riva_repository,
        "prepare_repository",
        lambda *_args: prepared.append(_args) or tmp_path / "repository",
    )
    monkeypatch.setattr(
        riva_repository.os,
        "execvpe",
        lambda *_args: pytest.fail("Riva must not serve"),
    )
    riva_repository.run()
    assert len(prepared) == 1
