# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint tests for ai-services/stt-server with a mocked ASR backend.

No GPU or NeMo: the module is imported via pythonpath and the backend's
``transcribe`` is replaced, so these run in CI. The GPU smoke test that
boots the real model lives in test_gpu_stt_server.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
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
