# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint tests for services/stt-server with a mocked ASR backend.

No GPU or NeMo: the module is imported via pythonpath and the backend's
``transcribe`` is replaced, so these run in CI. The GPU smoke test that
boots the real model lives in test_gpu_stt_server.py.
"""
from __future__ import annotations

from unittest.mock import Mock, call

import pytest
from fastapi.testclient import TestClient
from stt_server import __main__ as stt_main
from stt_server.__main__ import _build_app

_WAV = ("file", ("audio.wav", b"RIFF....WAVE", "audio/wav"))


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
