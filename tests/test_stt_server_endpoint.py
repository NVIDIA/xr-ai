# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint tests for services/stt-server with a mocked ASR backend.

No GPU or NeMo: the module is imported via pythonpath and the backend's
``transcribe`` is replaced, so these run in CI. The GPU smoke test that
boots the real model lives in test_gpu_stt_server.py.
"""
from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml
from fastapi.testclient import TestClient
from stt_server import __main__ as stt_main
from stt_server.__main__ import _build_app

_WAV = ("file", ("audio.wav", b"RIFF....WAVE", "audio/wav"))
_REPO_ROOT = Path(__file__).resolve().parents[1]
_STT_CONFIGS = (
    "services/stt-server/stt_server.yaml",
    "agent-samples/simple-vlm-example/yaml/stt_server.yaml",
    "agent-samples/xr-render-demo/yaml/stt_server.yaml",
    "agent-samples/model-servers/yaml/spark/stt_server.yaml",
    "agent-samples/model-servers/yaml/96G_blackwell/stt_server.yaml",
    "agent-samples/model-servers/yaml/dual_48G_ada/stt_server.yaml",
)


@pytest.fixture()
def app_and_backend(tmp_path):
    return _build_app({"model": "dummy"}, tmp_path)


def test_transcription_success(app_and_backend, monkeypatch):
    app, backend = app_and_backend
    monkeypatch.setattr(backend, "transcribe", lambda path: "hello world")
    with TestClient(app) as client:
        resp = client.post("/v1/audio/transcriptions", files=[_WAV])
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello world"}


def test_backend_failure_returns_500(app_and_backend, monkeypatch):
    """A backend exception surfaces as a 5xx with a generic detail: not an
    empty 200, and not the exception text (backend internals must not leak)."""
    app, backend = app_and_backend

    def _boom(path):
        raise RuntimeError("INTERNAL ASSERT FAILED at /internal/path CUDACachingAllocator")

    monkeypatch.setattr(backend, "transcribe", _boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/audio/transcriptions", files=[_WAV])
    assert resp.status_code == 500
    assert resp.json()["detail"] == "transcription failed"
    assert "CUDACachingAllocator" not in resp.text


def test_transcription_text_format_success(app_and_backend, monkeypatch):
    app, backend = app_and_backend
    monkeypatch.setattr(backend, "transcribe", lambda path: "hello world")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files=[_WAV],
            data={"response_format": "text"},
        )
    assert resp.status_code == 200
    assert resp.text == "hello world"
    assert resp.headers["content-type"].startswith("text/plain")


def test_default_startup_timeout_allows_cold_download():
    assert stt_main._DEFAULT_STARTUP_TIMEOUT_S == 600


@pytest.mark.parametrize("relative_path", _STT_CONFIGS)
def test_shipped_stt_configs_expose_startup_timeout(relative_path):
    config = yaml.safe_load((_REPO_ROOT / relative_path).read_text())
    assert config["startup_timeout_s"] == stt_main._DEFAULT_STARTUP_TIMEOUT_S


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf"],
)
def test_startup_timeout_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="finite number greater than zero"):
        stt_main._parse_startup_timeout(value)


def test_start_persistent_server_returns_healthy_child(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(stt_main.subprocess, "Popen", popen)
    monkeypatch.setattr(stt_main, "_health_url_ok", lambda _url: True)

    result = stt_main._start_persistent_server(
        ["stt", "--_serve"], "http://health", startup_timeout_s=600
    )

    assert result is process
    popen.assert_called_once_with(["stt", "--_serve"], start_new_session=True)
    process.wait.assert_not_called()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_startup_signal_terminates_child_and_restores_handlers(monkeypatch, signum):
    process = Mock()
    popen = Mock(return_value=process)
    terminate = Mock()
    original_handlers = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
    }
    installed_handlers = dict(original_handlers)

    def _signal(sig, handler):
        previous = installed_handlers[sig]
        installed_handlers[sig] = handler
        return previous

    def _interrupt_wait(*_args, **_kwargs):
        installed_handlers[signum](signum, None)

    monkeypatch.setattr(stt_main.subprocess, "Popen", popen)
    monkeypatch.setattr(stt_main.signal, "signal", _signal)
    monkeypatch.setattr(stt_main, "_wait_until_healthy", _interrupt_wait)
    monkeypatch.setattr(stt_main, "_terminate_process", terminate)

    with pytest.raises(SystemExit) as exc_info:
        stt_main._start_persistent_server(["stt", "--_serve"], "http://health", 600)

    assert exc_info.value.code == 128 + signum
    terminate.assert_called_once_with(process)
    assert installed_handlers == original_handlers


def test_signal_during_spawn_terminates_child_before_health_wait(monkeypatch):
    process = Mock()
    terminate = Mock()
    wait_until_healthy = Mock()
    original_handlers = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
    }
    installed_handlers = dict(original_handlers)

    def _signal(sig, handler):
        previous = installed_handlers[sig]
        installed_handlers[sig] = handler
        return previous

    def _popen(*_args, **_kwargs):
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process

    monkeypatch.setattr(stt_main.signal, "signal", _signal)
    monkeypatch.setattr(stt_main.subprocess, "Popen", _popen)
    monkeypatch.setattr(stt_main, "_wait_until_healthy", wait_until_healthy)
    monkeypatch.setattr(stt_main, "_terminate_process", terminate)

    with pytest.raises(SystemExit) as exc_info:
        stt_main._start_persistent_server(["stt", "--_serve"], "http://health", 600)

    assert exc_info.value.code == 128 + signal.SIGTERM
    terminate.assert_called_once_with(process)
    wait_until_healthy.assert_not_called()
    assert installed_handlers == original_handlers


def test_wait_until_healthy_reports_detached_child_exit(monkeypatch):
    process = Mock()
    process.poll.return_value = 17
    monkeypatch.setattr(
        stt_main,
        "_health_url_ok",
        lambda _url: pytest.fail("health should not be probed after child exit"),
    )

    with pytest.raises(RuntimeError, match="exited with status 17"):
        stt_main._wait_until_healthy(process, "http://health", timeout_s=600)

    process.wait.assert_not_called()


def test_idle_reports_detached_child_crash_without_health_delay(monkeypatch):
    process = Mock()
    process.wait.return_value = 23
    monkeypatch.setattr(
        stt_main,
        "_health_url_ok",
        lambda _url: pytest.fail("health should not delay child exit reporting"),
    )

    with pytest.raises(SystemExit, match="exited with status 23"):
        stt_main._idle_until_stopped("http://health", process, poll_s=5)

    process.wait.assert_called_once_with(timeout=5)


def test_idle_retries_transient_health_failures(monkeypatch):
    process = Mock()
    process.wait.side_effect = [
        stt_main.subprocess.TimeoutExpired("stt", 5),
        stt_main.subprocess.TimeoutExpired("stt", 5),
        stt_main.subprocess.TimeoutExpired("stt", 5),
        0,
    ]
    health = Mock(side_effect=[False, False, True])
    terminate = Mock()
    monkeypatch.setattr(stt_main, "_health_url_ok", health)
    monkeypatch.setattr(stt_main, "_terminate_process", terminate)

    stt_main._idle_until_stopped("http://health", process, poll_s=5)

    assert health.call_count == 3
    terminate.assert_not_called()


def test_idle_terminates_child_after_consecutive_health_failures(monkeypatch):
    process = Mock()
    process.wait.side_effect = [
        stt_main.subprocess.TimeoutExpired("stt", 5)
        for _ in range(stt_main._IDLE_HEALTH_FAILURE_LIMIT)
    ]
    terminate = Mock()
    monkeypatch.setattr(stt_main, "_health_url_ok", lambda _url: False)
    monkeypatch.setattr(stt_main, "_terminate_process", terminate)

    with pytest.raises(SystemExit, match="consecutive health checks; terminated"):
        stt_main._idle_until_stopped("http://health", process, poll_s=5)

    assert process.wait.call_count == stt_main._IDLE_HEALTH_FAILURE_LIMIT
    terminate.assert_called_once_with(process)


def test_startup_timeout_terminates_detached_child(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(stt_main.subprocess, "Popen", popen)

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("cold start exceeded budget")

    monkeypatch.setattr(stt_main, "_wait_until_healthy", _timeout)

    with pytest.raises(TimeoutError, match="cold start exceeded budget"):
        stt_main._start_persistent_server(["stt", "--_serve"], "http://health", 600)

    popen.assert_called_once_with(["stt", "--_serve"], start_new_session=True)
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=stt_main._PROCESS_STOP_TIMEOUT_S)


def test_terminate_process_escalates_to_kill():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [
        stt_main.subprocess.TimeoutExpired("stt", stt_main._PROCESS_STOP_TIMEOUT_S),
        0,
    ]

    stt_main._terminate_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [
        call(timeout=stt_main._PROCESS_STOP_TIMEOUT_S),
        call(),
    ]
