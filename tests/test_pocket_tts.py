# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Pocket TTS service wrapper."""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import wave
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import xr_ai_launcher._stack as launcher_stack
import xr_ai_vllm
import yaml

from _helpers_subprocess import pick_free_port

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = _REPO_ROOT / "services" / "pocket-tts"
_REFERENCE_CONFIG = _PROJECT / "pocket_tts_server.yaml"
_DEFAULT_PORT = 8105
_PROFILE_CONFIGS = (
    _REFERENCE_CONFIG,
    _REPO_ROOT / "agent-samples/model-servers/yaml/spark/pocket_tts_server.yaml",
    _REPO_ROOT / "agent-samples/model-servers/yaml/96G_blackwell/pocket_tts_server.yaml",
    _REPO_ROOT / "agent-samples/model-servers/yaml/dual_48G_ada/pocket_tts_server.yaml",
)


def _pocket_command(*args: str) -> list[str]:
    """Run the service module with the test environment's Python."""
    return [sys.executable, "-m", "pocket_tts_server", *args]


def _pocket_environment(log_root: Path, fake_dependencies: Path) -> dict[str, str]:
    """Expose the service source and lightweight fake model to a subprocess."""
    python_path = os.pathsep.join(
        (str(fake_dependencies), str(_PROJECT), os.environ.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return {
        **os.environ,
        "PYTHONPATH": python_path,
        "XR_AI_LOG_ROOT": str(log_root),
    }


def _write_fake_pocket_tts(root: Path) -> None:
    package = root / "pocket_tts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """class _Model:
    sample_rate = 24000
    has_voice_cloning = False

    def to(self, device):
        self.device = device
        return self

    def get_state_for_audio_prompt(self, voice):
        return {"voice": voice}


class TTSModel:
    @staticmethod
    def load_model(*, language):
        return _Model()
"""
    )


def _load_main_module():
    spec = importlib.util.spec_from_file_location(
        "pocket_tts_server_main",
        _PROJECT / "pocket_tts_server" / "__main__.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def reshape(self, *shape):
        return _FakeTensor(self._values.reshape(*shape))

    def clamp(self, minimum, maximum):
        return _FakeTensor(np.clip(self._values, minimum, maximum))

    def __mul__(self, scale):
        return _FakeTensor(self._values * scale)

    def short(self):
        return _FakeTensor(self._values.astype(np.int16))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeModel:
    sample_rate = 24000
    has_voice_cloning = True

    def __init__(self) -> None:
        self.generated: list[tuple[object, str]] = []
        self.device = None
        self.load_order = []

    def to(self, device):
        self.device = device
        self.load_order.append(("model", device))
        return self

    def get_state_for_audio_prompt(self, voice: str) -> object:
        self.load_order.append(("voice", self.device))
        return {"voice": voice, "device": self.device}

    def generate_audio(self, state: object, text: str) -> _FakeTensor:
        self.generated.append((state, text))
        return _FakeTensor([-2.0, -0.5, 0.5, 2.0])


def _loaded_backend(module, monkeypatch: pytest.MonkeyPatch, device="cpu"):
    model = _FakeModel()
    load_model = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "pocket_tts",
        SimpleNamespace(TTSModel=SimpleNamespace(load_model=load_model)),
    )
    backend = module._PocketTTSBackend("bill_boerst", "english", device)
    backend._ensure_loaded()
    return backend, model, load_model


@pytest.mark.parametrize("config_path", _PROFILE_CONFIGS)
async def test_configs_select_cc0_voice(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    assert config["voice"] == "bill_boerst"
    assert config["language"] == "english"
    assert config["port"] == 8105


async def test_backend_loads_selected_model_and_voice(monkeypatch) -> None:
    module = _load_main_module()
    backend, _model, load_model = _loaded_backend(module, monkeypatch)

    assert backend.ready
    assert backend.sample_rate == 24000
    load_model.assert_called_once_with(language="english")
    assert backend.device == "cpu"


@pytest.mark.parametrize("device,expected", [("cuda", "cuda:1"), ("cuda:0", "cuda:0")])
async def test_cuda_model_moves_before_loading_voice(monkeypatch, device, expected) -> None:
    module = _load_main_module()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 1, device_count=lambda: 2,
    )))
    backend, model, _load_model = _loaded_backend(module, monkeypatch, device)
    assert backend.device == expected
    assert model.load_order == [("model", expected), ("voice", expected)]
    assert backend._voice_state["device"] == expected
    assert backend.ready


@pytest.mark.parametrize("available,device,message", [
    (False, "cuda:0", "CUDA is unavailable"),
    (True, "cuda:2", "device index 2 is not visible"),
])
async def test_cuda_configuration_fails_without_cpu_fallback(monkeypatch, available, device, message) -> None:
    module = _load_main_module()
    load_model = Mock()
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=SimpleNamespace(load_model=load_model)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: 1,
    )))
    backend = module._PocketTTSBackend("bill_boerst", "english", device)
    with pytest.raises((RuntimeError, ValueError), match=message):
        backend._ensure_loaded()
    load_model.assert_not_called()
    assert not backend.ready
    assert backend.device is None


@pytest.mark.parametrize("device", [None, True, 0, "", "auto", "mps", "cuda:-1", "cuda:one", "cpu:0"])
async def test_rejects_invalid_device(device) -> None:
    module = _load_main_module()
    with pytest.raises(ValueError, match="'device' must be"):
        module._PocketTTSBackend("bill_boerst", "english", device)


async def test_blackwell_profile_selects_cuda() -> None:
    config = yaml.safe_load(_PROFILE_CONFIGS[2].read_text())
    assert config["device"] == "cuda:0"


async def test_cpu_only_torch_index_is_not_forced() -> None:
    import tomllib

    project = tomllib.loads((_PROJECT / "pyproject.toml").read_text())
    assert "torch" not in project["tool"]["uv"]["sources"]


@pytest.mark.parametrize("requested,actual", [("cpu", "cpu"), ("cuda", "cuda:1"), ("cuda:1", "cuda:1")])
async def test_reuse_accepts_matching_device(monkeypatch, requested, actual) -> None:
    module = _load_main_module()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(
        json.dumps({"status": "ok", "device": actual}).encode(),
    ))
    module._check_reused_device("http://localhost:8105/health", requested)


@pytest.mark.parametrize("actual", [None, "cpu", "cuda:1"])
async def test_cuda_reuse_rejects_legacy_or_wrong_device(monkeypatch, tmp_path, actual) -> None:
    module = _load_main_module()
    config_path = tmp_path / "pocket.yaml"
    config_path.write_text(yaml.safe_dump({"device": "cuda:0"}))
    ready = tmp_path / "ready"
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: True)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(
        json.dumps({"status": "ok", "device": actual}).encode(),
    ))
    monkeypatch.setattr(module.os, "execvpe", Mock(side_effect=AssertionError("must not replace listener")))
    monkeypatch.setattr(module.sys, "argv", [
        "pocket_tts_server", "--config", str(config_path), "--ready-file", str(ready),
    ])
    with pytest.raises(SystemExit, match="Stop the owned TTS service"):
        module.run()
    assert not ready.exists()


async def test_health_reports_actual_loaded_device(monkeypatch, tmp_path) -> None:
    module = _load_main_module()
    app, backend = module._build_app({"voice": "bill_boerst", "device": "cuda:0"}, tmp_path)
    assert backend._requested_device == "cuda:0"
    backend._model = _FakeModel()
    backend.device = "cuda:0"
    health = next(route.endpoint for route in app.routes if route.path == "/health")
    assert health() == {"status": "ok", "device": "cuda:0"}


async def test_backend_returns_wav_and_pcm(monkeypatch) -> None:
    module = _load_main_module()
    backend, model, _load_model = _loaded_backend(module, monkeypatch)

    pcm = backend.synthesize("Hello", "pcm")
    wav = backend.synthesize("Hello", "wav")

    expected_pcm = np.array([-32767, -16383, 16383, 32767], dtype=np.int16).tobytes()
    assert pcm == expected_pcm
    with wave.open(io.BytesIO(wav), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(4) == pcm
    assert len(model.generated) == 2


async def test_backend_returns_valid_empty_wav(monkeypatch) -> None:
    module = _load_main_module()
    backend, model, _load_model = _loaded_backend(module, monkeypatch)

    wav = backend.synthesize("   ")

    with wave.open(io.BytesIO(wav), "rb") as wav_file:
        assert wav_file.getnframes() == 0
        assert wav_file.getframerate() == 24000
    assert model.generated == []


async def test_backend_rejects_unknown_response_format(monkeypatch) -> None:
    module = _load_main_module()
    backend, _model, _load_model = _loaded_backend(module, monkeypatch)

    with pytest.raises(ValueError, match="response_format"):
        backend.synthesize("Hello", "mp3")


async def test_backend_rejects_unknown_voice_before_model_load(monkeypatch) -> None:
    module = _load_main_module()
    load_model = Mock()
    monkeypatch.setitem(
        sys.modules,
        "pocket_tts",
        SimpleNamespace(TTSModel=SimpleNamespace(load_model=load_model)),
    )
    backend = module._PocketTTSBackend("bill_boerts", "english")

    with pytest.raises(ValueError, match="unsupported Pocket TTS voice"):
        backend._ensure_loaded()

    load_model.assert_not_called()


async def test_cancelled_queued_synthesis_skips_generation(monkeypatch) -> None:
    module = _load_main_module()
    backend, model, _load_model = _loaded_backend(module, monkeypatch)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(module._SynthesisCancelled):
        backend.synthesize("abandoned sentence", cancelled=cancelled)

    assert model.generated == []


async def test_http_contract_and_error_mapping(monkeypatch, tmp_path) -> None:
    from fastapi import HTTPException

    module = _load_main_module()
    app, backend = module._build_app({"voice": "bill_boerst"}, tmp_path)
    routes = {route.path: route for route in app.routes}

    async def run_inline(_executor, function, *args):
        return function(*args)

    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", run_inline)

    with pytest.raises(HTTPException) as health_error:
        routes["/health"].endpoint()
    assert health_error.value.status_code == 503

    backend._model = SimpleNamespace(sample_rate=24000)
    backend.synthesize = Mock(return_value=b"RIFFtest")
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    speech_request = SimpleNamespace(input="Hello", response_format="wav")
    response = await routes["/v1/audio/speech"].endpoint(speech_request, request)
    assert response.status_code == 200
    assert response.body == b"RIFFtest"
    assert response.media_type == "audio/wav"

    backend.synthesize.side_effect = ValueError("unsupported format")
    speech_request.response_format = "mp3"
    with pytest.raises(HTTPException) as format_error:
        await routes["/v1/audio/speech"].endpoint(speech_request, request)
    assert format_error.value.status_code == 400
    assert format_error.value.detail == "unsupported format"


async def test_scopes_hugging_face_cache_to_model_cache(tmp_path, monkeypatch) -> None:
    module = _load_main_module()
    model_cache = tmp_path / "models"
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.setitem(module.sys.modules, "uvicorn", Mock())

    def check_environment(_cfg: dict, resolved_cache: Path):
        assert resolved_cache == model_cache
        assert os.environ["HF_HOME"] == str(model_cache / "pocket" / "huggingface")
        assert os.environ["HF_XET_CACHE"] == str(model_cache / "pocket" / "xet")
        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
        raise RuntimeError("environment checked")

    monkeypatch.setattr(module, "_build_app", check_environment)
    with pytest.raises(RuntimeError, match="environment checked"):
        await module._run(
            {"voice": "bill_boerst", "model_cache": str(model_cache)},
            tmp_path,
        )


async def test_reuses_healthy_persistent_server(tmp_path, monkeypatch) -> None:
    module = _load_main_module()
    ready_file = tmp_path / "ready"
    monitor = Mock()
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: True)
    monkeypatch.setattr(module, "_monitor_reused_server", monitor)
    monkeypatch.delenv(module._READY_PROCESS_MAY_EXIT_ENV, raising=False)
    monkeypatch.setattr(
        module.os,
        "execvpe",
        Mock(side_effect=AssertionError("must not launch another server")),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["pocket_tts_server", "--ready-file", str(ready_file)],
    )

    module.run()

    assert ready_file.exists()
    monitor.assert_called_once_with("http://127.0.0.1:8105/health")


async def test_rejects_unhealthy_listener(tmp_path, monkeypatch) -> None:
    module = _load_main_module()
    ready_file = tmp_path / "ready"
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: False)
    monkeypatch.setattr(module, "_port_open", lambda _host, _port: True)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["pocket_tts_server", "--ready-file", str(ready_file)],
    )

    with pytest.raises(SystemExit, match="already in use"):
        module.run()
    assert not ready_file.exists()


async def test_model_load_timeout_does_not_wait_for_loader(tmp_path, monkeypatch) -> None:
    module = _load_main_module()
    release = threading.Event()
    backend = Mock()
    backend._ensure_loaded.side_effect = lambda: release.wait(timeout=1)
    monkeypatch.setitem(module.sys.modules, "uvicorn", Mock())
    monkeypatch.setattr(
        module,
        "_build_app",
        lambda _cfg, _model_cache: (Mock(), backend),
    )

    try:
        with pytest.raises(TimeoutError, match="within 0.01 seconds"):
            await module._run(
                {"voice": "bill_boerst", "startup_timeout_s": 0.01},
                tmp_path,
            )
    finally:
        release.set()


async def test_execs_managed_server_in_place(tmp_path, monkeypatch) -> None:
    module = _load_main_module()
    ready_file = tmp_path / "ready"
    execvpe = Mock()
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: False)
    monkeypatch.setattr(module, "_port_open", lambda _host, _port: False)
    monkeypatch.setattr(module, "_ensure_owned_process_group", lambda: 8105)
    monkeypatch.setattr(module.os, "execvpe", execvpe)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["pocket_tts_server", "--ready-file", str(ready_file)],
    )

    module.run()

    executable, argv, child_env = execvpe.call_args.args
    assert executable == module.sys.executable
    assert argv[:4] == [module.sys.executable, "-m", "pocket_tts_server", "--_serve"]
    assert argv[-2:] == ["--ready-file", str(ready_file)]
    assert child_env["XR_AI_VLLM_MANAGED"] == "1"
    assert child_env["XR_AI_VLLM_PORT"] == "8105"
    assert child_env[module._PROCESS_GROUP_ENV] == "8105"
    assert not ready_file.exists()


async def test_reuse_exits_when_launcher_allows_ready_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    ready_file = tmp_path / "ready"
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: True)
    monkeypatch.setattr(
        module,
        "_monitor_reused_server",
        lambda _url: pytest.fail("ready-exit launch must not remain monitored"),
    )
    monkeypatch.setenv(module._READY_PROCESS_MAY_EXIT_ENV, "1")
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["pocket_tts_server", "--ready-file", str(ready_file)],
    )

    module.run()

    assert ready_file.exists()


async def test_reuse_monitor_tolerates_transient_health_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    health = Mock(side_effect=[False, True, False, False, False])
    monkeypatch.setattr(module, "_health_url_ok", health)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit, match="3 consecutive health checks"):
        module._monitor_reused_server("http://127.0.0.1:8105/health")

    assert health.call_count == 5


async def test_probes_configured_non_loopback_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    config_path = tmp_path / "pocket.yaml"
    config_path.write_text(yaml.safe_dump({"host": "192.0.2.10", "port": 8123}))
    health = Mock(return_value=False)
    port_open = Mock(return_value=False)
    execvpe = Mock()
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", health)
    monkeypatch.setattr(module, "_port_open", port_open)
    monkeypatch.setattr(module, "_ensure_owned_process_group", lambda: 8123)
    monkeypatch.setattr(module.os, "execvpe", execvpe)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["pocket_tts_server", "--config", str(config_path)],
    )

    module.run()

    expected_health_url = "http://192.0.2.10:8123/health"
    health.assert_called_once_with(expected_health_url)
    port_open.assert_called_once_with("192.0.2.10", 8123)
    assert execvpe.call_args.args[1][-2:] == ["--config", str(config_path)]
    assert execvpe.call_args.args[2][module._PROCESS_GROUP_ENV] == "8123"


@pytest.mark.parametrize(
    ("bind_host", "expected_probe_host", "expected_health_url"),
    (
        ("0.0.0.0", "127.0.0.1", "http://127.0.0.1:8105/health"),
        ("::", "::1", "http://[::1]:8105/health"),
    ),
)
async def test_normalizes_wildcard_probe_hosts(
    bind_host: str,
    expected_probe_host: str,
    expected_health_url: str,
) -> None:
    module = _load_main_module()
    probe_host = module._probe_host(bind_host)

    assert probe_host == expected_probe_host
    assert module._health_url(probe_host, 8105) == expected_health_url


async def test_startup_timeout_cancels_server_that_never_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    ready_file = tmp_path / "ready"
    backend = Mock()

    class NeverStartedServer:
        started = False
        cancelled = False

        async def serve(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    server = NeverStartedServer()
    uvicorn = Mock()
    uvicorn.Server.return_value = server
    monkeypatch.setitem(module.sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(
        module,
        "_build_app",
        lambda _cfg, _model_cache: (Mock(), backend),
    )

    with pytest.raises(TimeoutError, match="within 0.01 seconds"):
        await module._run(
            {"voice": "bill_boerst", "startup_timeout_s": 0.01},
            tmp_path,
            ready_file,
        )

    assert server.cancelled
    assert not ready_file.exists()


@pytest.mark.parametrize("error", [TimeoutError("timed out"), OSError("bind failed")])
async def test_serve_translates_startup_errors(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()

    def fail(coroutine) -> None:
        coroutine.close()
        raise error

    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module.asyncio, "run", fail)
    monkeypatch.setattr(module.sys, "argv", ["pocket_tts_server", "--_serve"])

    with pytest.raises(SystemExit, match=rf"\[pocket_tts_server\] {error}") as exc:
        module.run()

    assert exc.value.__cause__ is None


async def test_uses_existing_dedicated_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 1234)

    assert module._ensure_owned_process_group() == 1234


async def test_preserves_launcher_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 4321)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 4321)
    monkeypatch.setenv(module._LAUNCHER_GROUP_OWNER_ENV, module._LAUNCHER_GROUP_OWNER)

    assert module._ensure_owned_process_group() == 4321


async def test_does_not_claim_inherited_non_launcher_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 4321)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 4321)
    monkeypatch.delenv(module._LAUNCHER_GROUP_OWNER_ENV, raising=False)

    assert module._ensure_owned_process_group() is None


async def test_does_not_claim_unisolated_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 4321)

    assert module._ensure_owned_process_group() is None


async def test_omits_group_marker_when_ownership_is_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_main_module()
    execvpe = Mock()
    monkeypatch.setenv(module._PROCESS_GROUP_ENV, "4321")
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: False)
    monkeypatch.setattr(module, "_port_open", lambda _host, _port: False)
    monkeypatch.setattr(module, "_ensure_owned_process_group", lambda: None)
    monkeypatch.setattr(module.os, "execvpe", execvpe)
    monkeypatch.setattr(module.sys, "argv", ["pocket_tts_server"])

    module.run()

    assert module._PROCESS_GROUP_ENV not in execvpe.call_args.args[2]


class _ServerExited(Exception):
    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output
        super().__init__(f"pocket_tts_server exited early with code {returncode}")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        try:
            connection.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


async def _wait_for_ready_file(
    ready_file: Path,
    *,
    proc: subprocess.Popen,
    timeout: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if ready_file.exists():
            return
        if proc.poll() is not None:
            output = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise _ServerExited(proc.returncode, output)
        await asyncio.sleep(0.05)
    raise TimeoutError(
        f"pocket_tts_server did not signal ready within {timeout} seconds"
    )


@pytest.mark.integration
async def test_launcher_abort_force_kills_pocket_in_its_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "pocket_tts_server"
    executable.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
import signal
import time
from pathlib import Path

from pocket_tts_server.__main__ import _ensure_owned_process_group

parser = argparse.ArgumentParser()
parser.add_argument("--ready-file", required=True)
args = parser.parse_args()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(args.ready_file).write_text(json.dumps({
    "pid": os.getpid(),
    "group": _ensure_owned_process_group(),
    "session": os.getsid(0),
}))
while True:
    time.sleep(1)
"""
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    python_path = os.pathsep.join(
        (str(_PROJECT), os.environ.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    monkeypatch.setenv("PYTHONPATH", python_path)

    ready_file = tmp_path / "abort.ready"
    process = launcher_stack._spawn(
        launcher_stack.Process("tts", _REPO_ROOT / "tests", "pocket_tts_server"),
        _REPO_ROOT,
        ready_file,
    )

    child_pid: int | None = None
    try:
        await _wait_for_ready_file(ready_file, proc=process, timeout=20)
        state = json.loads(ready_file.read_text())
        child_pid = state["pid"]
        assert state["group"] == process.pid
        assert state["session"] == process.pid

        monkeypatch.setattr(launcher_stack, "_STOP_TIMEOUT", 0.1)
        launcher_stack._shutdown({"tts": process})

        deadline = asyncio.get_running_loop().time() + 5
        while (
            xr_ai_vllm._docker.process_group_alive(process.pid)
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
        assert not xr_ai_vllm._docker.process_group_alive(process.pid)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=5)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.integration
async def test_managed_reuse_and_process_group_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_dependencies = tmp_path / "fake-dependencies"
    _write_fake_pocket_tts(fake_dependencies)
    port = pick_free_port(_DEFAULT_PORT)
    cfg_path = tmp_path / "pocket_tts_server.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "voice": "bill_boerst",
                "port": port,
                "host": "127.0.0.1",
                "startup_timeout_s": 30,
                "model_cache": str(tmp_path / "models"),
            }
        )
    )
    env = _pocket_environment(tmp_path / "logs", fake_dependencies)
    env["_XR_AI_LAUNCHER_PROCESS_GROUP_OWNER"] = "pocket_tts_server"
    env["_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT"] = "1"
    owner_ready = tmp_path / "owner.ready"
    reuse_ready = tmp_path / "reuse.ready"
    owner: subprocess.Popen | None = None
    reuse: subprocess.Popen | None = None

    try:
        owner = subprocess.Popen(
            _pocket_command("--config", str(cfg_path), "--ready-file", str(owner_ready)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        await _wait_for_ready_file(owner_ready, proc=owner, timeout=20)

        reuse = subprocess.Popen(
            _pocket_command("--config", str(cfg_path), "--ready-file", str(reuse_ready)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        await _wait_for_ready_file(reuse_ready, proc=reuse, timeout=10)
        assert reuse.wait(timeout=5) == 0
        assert owner.poll() is None

        monkeypatch.setattr(
            xr_ai_vllm._docker,
            "container_on_port_checked",
            lambda _port: (None, True),
        )
        listener_pid, inspected, listening = xr_ai_vllm._docker.pid_on_port_checked(port)
        assert inspected and listening and listener_pid is not None
        assert (
            xr_ai_vllm._docker._pocket_owned_process_group(listener_pid, port)
            == owner.pid
        )
        stopped = await asyncio.get_running_loop().run_in_executor(
            None,
            xr_ai_vllm.stop_persistent_servers,
            [("tts", port)],
        )
        assert stopped
        assert owner.wait(timeout=5) is not None
        assert not _port_open(port)
        with pytest.raises(ProcessLookupError):
            os.killpg(owner.pid, 0)
    finally:
        for process in (reuse, owner):
            if process is None or process.poll() is not None:
                continue
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
