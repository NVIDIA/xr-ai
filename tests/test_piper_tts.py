# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for services/piper-tts.

Spawns the Piper TTS server as a subprocess (out of its own venv — the
test harness must not import the heavy piper/fastapi deps) and round-trips
a tiny synthesis request through the OpenAI-compatible
``POST /v1/audio/speech`` endpoint.

CPU-only: Piper runs ONNX on CPU at ~100 ms/sentence. No ``gpu`` marker,
so CI picks this up. Skipped cleanly when the environment can't support it:
``uv`` is missing, the piper venv hasn't been ``uv sync``'d, or the
configured voice can't be obtained (offline with an empty cache, or a
transient HuggingFace download failure — the server signals this with a
dedicated exit code). Any other early exit fails the test with the server's
captured output so the cause is visible in CI.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import signal
import socket  # used by local _port_open below; _pick_port moved to _helpers_subprocess but _port_open stays here
import subprocess
import threading
import urllib.request
from contextlib import suppress
from pathlib import Path
from unittest.mock import Mock

import pytest
import xr_ai_launcher._stack as launcher_stack
import xr_ai_vllm
import yaml

from _helpers_subprocess import pick_free_port

pytestmark = pytest.mark.asyncio


_REPO_ROOT     = Path(__file__).resolve().parents[1]
_PIPER_PROJECT = _REPO_ROOT / "services" / "piper-tts"
_PIPER_YAML    = _PIPER_PROJECT / "piper_tts_server.yaml"
_DEFAULT_PORT  = 8105
_PIPER_CONFIGS = (
    _PIPER_YAML,
    _REPO_ROOT / "agent-samples/model-servers/yaml/spark/piper_tts_server.yaml",
    _REPO_ROOT / "agent-samples/model-servers/yaml/96G_blackwell/piper_tts_server.yaml",
    _REPO_ROOT / "agent-samples/model-servers/yaml/dual_48G_ada/piper_tts_server.yaml",
)

# Must match _EXIT_VOICE_UNAVAILABLE in piper_tts_server/__main__.py: the
# server uses this exit code when the voice can't be obtained for
# environmental reasons (offline empty cache / transient HF download failure),
# which the smoke test treats as skip rather than fail.
_EXIT_VOICE_UNAVAILABLE = 3


def _piper_command(*args: str) -> list[str]:
    """Run Piper through uv without assuming a project-local environment."""
    return [
        "uv",
        "run",
        "--quiet",
        "--project",
        str(_PIPER_PROJECT),
        "piper_tts_server",
        *args,
    ]


def _piper_environment(log_root: Path) -> dict[str, str]:
    """Keep Piper's environment independent from the active test venv."""
    environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(_PIPER_PROJECT / ".venv"),
        "XR_AI_LOG_ROOT": str(log_root),
    }
    environment.pop("VIRTUAL_ENV", None)
    return environment


def _load_piper_main_module():
    spec = importlib.util.spec_from_file_location(
        "piper_tts_server_main",
        _PIPER_PROJECT / "piper_tts_server" / "__main__.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_piper_config_parses_quoted_use_cuda_false() -> None:
    module = _load_piper_main_module()
    assert module._parse_config_bool("false", "use_cuda") is False


async def test_piper_config_rejects_unknown_use_cuda_string() -> None:
    module = _load_piper_main_module()
    with pytest.raises(ValueError, match="use_cuda"):
        module._parse_config_bool("sometimes", "use_cuda")


async def test_piper_scopes_xet_cache_to_model_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    model_cache = tmp_path / "models"
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.setitem(module.sys.modules, "uvicorn", Mock())

    def check_environment(_cfg: dict, resolved_cache: Path) -> None:
        assert resolved_cache == model_cache
        assert os.environ["HF_XET_CACHE"] == str(model_cache / "piper" / "xet")
        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
        raise RuntimeError("environment checked")

    monkeypatch.setattr(module, "_build_app", check_environment)

    with pytest.raises(RuntimeError, match="environment checked"):
        await module._run(
            {"voice": "en_US-lessac-medium", "model_cache": str(model_cache)},
            tmp_path,
        )


@pytest.mark.parametrize("config_path", _PIPER_CONFIGS)
async def test_piper_configs_allow_cold_start(config_path: Path) -> None:
    module = _load_piper_main_module()
    config = yaml.safe_load(config_path.read_text())
    assert config["startup_timeout_s"] == module._DEFAULT_STARTUP_TIMEOUT_S


async def test_piper_reuses_healthy_persistent_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    ready_file = tmp_path / "ready"
    execvpe = Mock(side_effect=AssertionError("must not launch a second server"))
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: True)
    monkeypatch.setattr(module.os, "execvpe", execvpe)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["piper_tts_server", "--ready-file", str(ready_file)],
    )

    module.run()

    assert ready_file.exists()
    execvpe.assert_not_called()


async def test_piper_probes_configured_non_loopback_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    config_path = tmp_path / "piper.yaml"
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
        ["piper_tts_server", "--config", str(config_path)],
    )

    module.run()

    expected_health_url = "http://192.0.2.10:8123/health"
    health.assert_called_once_with(expected_health_url)
    port_open.assert_called_once_with("192.0.2.10", 8123)
    execvpe.assert_called_once()
    assert execvpe.call_args.args[1][-2:] == ["--config", str(config_path)]
    assert execvpe.call_args.args[2][module._PROCESS_GROUP_ENV] == "8123"


@pytest.mark.parametrize(
    ("bind_host", "expected_probe_host", "expected_health_url"),
    (
        ("0.0.0.0", "127.0.0.1", "http://127.0.0.1:8105/health"),
        ("::", "::1", "http://[::1]:8105/health"),
    ),
)
async def test_piper_normalizes_wildcard_probe_hosts(
    bind_host: str,
    expected_probe_host: str,
    expected_health_url: str,
) -> None:
    module = _load_piper_main_module()
    probe_host = module._probe_host(bind_host)

    assert probe_host == expected_probe_host
    assert module._health_url(probe_host, 8105) == expected_health_url


async def test_piper_rejects_unhealthy_listener_without_signaling_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    ready_file = tmp_path / "ready"
    execvpe = Mock(side_effect=AssertionError("must not launch into an occupied port"))
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: False)
    monkeypatch.setattr(module, "_port_open", lambda _host, _port: True)
    monkeypatch.setattr(module.os, "execvpe", execvpe)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["piper_tts_server", "--ready-file", str(ready_file)],
    )

    with pytest.raises(SystemExit, match="port 8105 is already in use"):
        module.run()

    assert not ready_file.exists()
    execvpe.assert_not_called()


async def test_piper_model_load_timeout_does_not_wait_for_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
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
                {"voice": "en_US-lessac-medium", "startup_timeout_s": 0.01},
                tmp_path,
            )
    finally:
        release.set()


async def test_piper_startup_timeout_cancels_server_that_never_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
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
            {"voice": "en_US-lessac-medium", "startup_timeout_s": 0.01},
            tmp_path,
            ready_file,
        )

    assert server.cancelled
    assert not ready_file.exists()


@pytest.mark.parametrize("error", [TimeoutError("timed out"), OSError("bind failed")])
async def test_piper_serve_translates_startup_errors(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()

    def fail(coroutine) -> None:
        coroutine.close()
        raise error

    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module.asyncio, "run", fail)
    monkeypatch.setattr(module.sys, "argv", ["piper_tts_server", "--_serve"])

    with pytest.raises(SystemExit, match=rf"\[piper_tts_server\] {error}") as exc:
        module.run()

    assert exc.value.__cause__ is None


async def test_piper_uses_existing_dedicated_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 1234)
    monkeypatch.setattr(
        module.os,
        "setsid",
        lambda: pytest.fail("an existing dedicated session must be preserved"),
    )

    assert module._ensure_owned_process_group() == 1234


async def test_piper_preserves_launcher_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 4321)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 4321)
    monkeypatch.setenv(
        module._LAUNCHER_GROUP_OWNER_ENV,
        module._LAUNCHER_GROUP_OWNER,
    )
    monkeypatch.setattr(
        module.os,
        "setsid",
        lambda: pytest.fail("Piper must remain reachable by launcher SIGKILL"),
    )

    assert module._ensure_owned_process_group() == 4321


async def test_piper_does_not_claim_inherited_non_launcher_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 4321)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 4321)
    monkeypatch.delenv(module._LAUNCHER_GROUP_OWNER_ENV, raising=False)
    monkeypatch.setattr(
        module.os,
        "setsid",
        lambda: pytest.fail("Piper must not change a caller's process topology"),
    )

    assert module._ensure_owned_process_group() is None


async def test_piper_does_not_claim_an_unisolated_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1234)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(module.os, "getsid", lambda _pid: 4321)

    assert module._ensure_owned_process_group() is None


async def test_piper_execs_managed_server_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
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
        ["piper_tts_server", "--ready-file", str(ready_file)],
    )

    module.run()

    executable, argv, child_env = execvpe.call_args.args
    assert executable == module.sys.executable
    assert argv[:4] == [
        module.sys.executable,
        "-m",
        "piper_tts_server",
        "--_serve",
    ]
    assert argv[-2:] == ["--ready-file", str(ready_file)]
    assert child_env["XR_AI_VLLM_MANAGED"] == "1"
    assert child_env["XR_AI_VLLM_PORT"] == "8105"
    assert child_env[module._PROCESS_GROUP_ENV] == "8105"
    assert not ready_file.exists()


async def test_piper_omits_group_marker_when_ownership_is_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_piper_main_module()
    execvpe = Mock()
    monkeypatch.setenv(module._PROCESS_GROUP_ENV, "4321")
    monkeypatch.setattr(module, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(module, "_health_url_ok", lambda _url: False)
    monkeypatch.setattr(module, "_port_open", lambda _host, _port: False)
    monkeypatch.setattr(module, "_ensure_owned_process_group", lambda: None)
    monkeypatch.setattr(module.os, "execvpe", execvpe)
    monkeypatch.setattr(module.sys, "argv", ["piper_tts_server"])

    module.run()

    assert module._PROCESS_GROUP_ENV not in execvpe.call_args.args[2]


@pytest.mark.integration
async def test_launcher_abort_force_kills_piper_in_its_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")

    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "piper_tts_server"
    executable.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
import signal
import time
from pathlib import Path

from piper_tts_server.__main__ import _ensure_owned_process_group

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
    monkeypatch.setenv(
        "PATH",
        f"{executable_dir}{os.pathsep}{os.environ['PATH']}",
    )
    python_path = os.environ.get("PYTHONPATH")
    piper_python_path = str(_PIPER_PROJECT)
    if python_path:
        piper_python_path += os.pathsep + python_path
    monkeypatch.setenv("PYTHONPATH", piper_python_path)

    ready_file = tmp_path / "abort.ready"
    process = launcher_stack._spawn(
        launcher_stack.Process(
            "tts",
            _REPO_ROOT / "tests",
            "piper_tts_server",
        ),
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


class _ServerExited(Exception):
    """Raised when piper_tts_server exits before binding its port.

    Carries the process return code and the captured stdout+stderr so callers
    can decide whether to skip (environmental) or fail (real error) — and so
    the real cause is visible instead of a bare "exited with code N".
    """

    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output
        super().__init__(f"piper_tts_server exited early with code {returncode}")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _voice_cached(voice: str, hf_cache: Path) -> bool:
    """True iff both the .onnx and .onnx.json for *voice* are present in cache.

    Walks the HF cache layout directly so we don't pull huggingface_hub into
    the test venv just for an offline probe.
    """
    parts = voice.split("-")
    if len(parts) < 3:
        return False
    locale, speaker, quality = parts[0], parts[1], "-".join(parts[2:])
    lang = locale.split("_")[0]
    rel  = f"{lang}/{locale}/{speaker}/{quality}/{voice}"
    snapshots = hf_cache / "models--rhasspy--piper-voices" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        (snap / f"{rel}.onnx").exists() and (snap / f"{rel}.onnx.json").exists()
        for snap in snapshots.iterdir()
    )


async def _wait_for_port(port: int, *, proc: subprocess.Popen, timeout: float) -> None:
    """Poll the bind port until it accepts a TCP connection.

    Raises ``_ServerExited`` (with the captured output) if the process dies
    before binding, or ``TimeoutError`` if the port never opens.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if proc.poll() is not None:
            # The process is dead, so its pipe won't block — read the captured
            # stdout+stderr and surface it instead of just the exit code.
            output = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise _ServerExited(proc.returncode, output)
        if _port_open(port):
            return
        await asyncio.sleep(0.2)
    raise TimeoutError(f"piper_tts_server did not open port {port} within {timeout}s")


async def _wait_for_ready_file(
    ready_file: Path,
    *,
    proc: subprocess.Popen,
    timeout: float,
) -> None:
    """Wait for launcher readiness while preserving early process diagnostics."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if ready_file.exists():
            return
        if proc.poll() is not None:
            output = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise _ServerExited(proc.returncode, output)
        await asyncio.sleep(0.05)
    raise TimeoutError(
        f"piper_tts_server did not signal ready within {timeout}s"
    )


def _post_speech(port: int, voice: str, text: str) -> bytes:
    payload = json.dumps({
        "model":           voice,
        "input":           text,
        "voice":           voice,
        "response_format": "wav",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        assert resp.status == 200, f"unexpected status {resp.status}"
        return resp.read()


async def test_piper_tts_smoke(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")

    ref_cfg     = yaml.safe_load(_PIPER_YAML.read_text())
    voice       = ref_cfg["voice"]
    model_cache = (_PIPER_YAML.parent / ref_cfg.get("model_cache", "../../models")).resolve()
    # The piper server eagerly resolves + downloads the configured voice
    # on startup (see services/piper-tts/piper_tts_server/__main__.py),
    # so we don't pre-check voice cache state here.

    port = pick_free_port(_DEFAULT_PORT)

    cfg_path = tmp_path / "piper_tts_server.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "voice":       voice,
        "port":        port,
        "host":        "127.0.0.1",
        "use_cuda":    False,
        "model_cache": str(model_cache),
    }))

    env = _piper_environment(tmp_path / "logs")

    proc = subprocess.Popen(
        _piper_command("--_serve", "--config", str(cfg_path)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,  # SIGTERM must reach the whole tree
    )

    try:
        # First-run voice download (~50–200 MB) plus ONNX init can take a
        # couple of minutes on a cold cache; reuse is sub-second.
        try:
            await _wait_for_port(port, proc=proc, timeout=300.0)
        except _ServerExited as exc:
            tail = exc.output.strip()[-2000:]
            # A voice-unavailable exit is environmental, not a code bug: an
            # offline empty cache or a transient HuggingFace download failure
            # (HTTP 429 rate-limit on the anonymous voice download). Skip
            # cleanly rather than fail the suite — the smoke test only asserts
            # the server path when the voice can actually be obtained.
            if exc.returncode == _EXIT_VOICE_UNAVAILABLE:
                pytest.skip(
                    "piper voice could not be obtained (offline cache or "
                    f"transient HuggingFace download failure):\n{tail}"
                )
            # Any other early exit is a real failure — surface the captured
            # server output so it's diagnosable from the CI log.
            pytest.fail(
                f"piper_tts_server exited with code {exc.returncode}:\n{tail}"
            )
        body = await asyncio.get_running_loop().run_in_executor(
            None, _post_speech, port, voice, "Hello, world.",
        )
        # Regression #194: whitespace-only input must return a valid (empty)
        # WAV, not an HTTP 500 (wave.Error: # channels not specified).
        # _post_speech asserts status 200, so a 500 here fails the test.
        empty_body = await asyncio.get_running_loop().run_in_executor(
            None, _post_speech, port, voice, "   ",
        )
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    assert body, "empty response body"
    assert body[:4] == b"RIFF", f"expected WAV RIFF header, got {body[:8]!r}"
    # Whitespace-only input → a valid WAV header (zero audio frames), not a 500.
    assert empty_body[:4] == b"RIFF", (
        f"expected WAV for whitespace input, got {empty_body[:8]!r}"
    )


async def test_piper_managed_reuse_and_process_group_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")

    ref_cfg = yaml.safe_load(_PIPER_YAML.read_text())
    model_cache = (
        _PIPER_YAML.parent / ref_cfg.get("model_cache", "../../models")
    ).resolve()
    port = pick_free_port(_DEFAULT_PORT)
    cfg_path = tmp_path / "piper_tts_server.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "voice": ref_cfg["voice"],
        "port": port,
        "host": "127.0.0.1",
        "use_cuda": False,
        "startup_timeout_s": 300,
        "model_cache": str(model_cache),
    }))
    env = _piper_environment(tmp_path / "logs")
    env["_XR_AI_LAUNCHER_PROCESS_GROUP_OWNER"] = "piper_tts_server"
    owner_ready = tmp_path / "owner.ready"
    reuse_ready = tmp_path / "reuse.ready"
    owner: subprocess.Popen | None = None
    reuse: subprocess.Popen | None = None

    try:
        owner = subprocess.Popen(
            _piper_command(
                "--config", str(cfg_path), "--ready-file", str(owner_ready)
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            await _wait_for_ready_file(owner_ready, proc=owner, timeout=300)
        except _ServerExited as exc:
            tail = exc.output.strip()[-2000:]
            if exc.returncode == _EXIT_VOICE_UNAVAILABLE:
                pytest.skip(
                    "piper voice could not be obtained (offline cache or "
                    f"transient HuggingFace download failure):\n{tail}"
                )
            pytest.fail(
                f"piper_tts_server exited with code {exc.returncode}:\n{tail}"
            )

        reuse = subprocess.Popen(
            _piper_command(
                "--config", str(cfg_path), "--ready-file", str(reuse_ready)
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        await _wait_for_ready_file(reuse_ready, proc=reuse, timeout=10)
        assert reuse.wait(timeout=5) == 0
        assert owner.poll() is None

        # Docker discovery is independent of this local-process lifecycle test.
        monkeypatch.setattr(
            xr_ai_vllm._docker,
            "container_on_port_checked",
            lambda _port: (None, True),
        )
        listener_pid, inspected, listening = (
            xr_ai_vllm._docker.pid_on_port_checked(port)
        )
        assert inspected and listening and listener_pid is not None
        assert (
            xr_ai_vllm._docker._piper_owned_process_group(listener_pid, port)
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
