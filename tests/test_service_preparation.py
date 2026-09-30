# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


magpie = _load_main("magpie-tts", "magpie_tts_server")
pocket = _load_main("pocket-tts", "pocket_tts_server")


@pytest.fixture(autouse=True)
def _stable_cache_env(monkeypatch):
    for name in ("HF_HOME", "HF_XET_CACHE", "HF_XET_HIGH_PERFORMANCE", "NEMO_CACHE_DIR"):
        monkeypatch.setenv(name, "/test/cache")


@pytest.mark.parametrize("backend", ["pip", "docker"])
def test_vllm_prepare_downloads_with_selected_backend_and_never_serves(
    backend, tmp_path, monkeypatch,
):
    calls: list[str] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(_pip, "prepare", lambda _model: calls.append("pip"))
    monkeypatch.setattr(_docker, "_maybe_ngc_login", lambda _image: None)
    monkeypatch.setattr(_docker.subprocess, "run", lambda command, **_: commands.append(command))
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
        command = commands[0]
        payload = command[command.index("-c") + 1]
        assert payload.endswith("true")
        assert "snapshot_download" in payload
        assert "TQDM_POSITION=-1" in command
        assert command.count("--init") == 1


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


@pytest.mark.parametrize("module", [stt, magpie])
def test_native_model_prepare_downloads_without_loading_server(
    module, tmp_path, monkeypatch,
):
    downloads: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda *, repo_id: downloads.append(repo_id),
        ),
    )
    module._prepare({"model": "org/model"}, tmp_path)
    assert downloads == ["org/model"]


def test_pocket_prepare_forces_cpu_and_loads_artifacts(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str] | str] = []
    def backend(voice, language, device):
        calls.append((voice, language, device))
        return SimpleNamespace(_ensure_loaded=lambda: calls.append("loaded"))
    monkeypatch.setattr(pocket, "_PocketTTSBackend", backend)
    pocket._prepare({"voice": "speaker"}, tmp_path)
    assert calls == [("speaker", "english", "cpu"), "loaded"]


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
