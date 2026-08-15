# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for full-frame camera capture in the basic web client."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "client-samples/web/StreamKit/Backends/LiveKit/LiveKitBackend.js"
_SESSION = _ROOT / "client-samples/web/StreamKit/StreamSession.js"
_APP = _ROOT / "client-samples/web/App/app.js"
_PAGE = _ROOT / "client-samples/web/index.html"


def test_camera_capture_does_not_force_livekit_h720_constraints() -> None:
    backend = _BACKEND.read_text(encoding="utf-8")

    assert "createLocalVideoTrack" not in backend
    assert "navigator.mediaDevices.getUserMedia" in backend
    assert "resizeMode: 'none'" in backend
    assert "publishTrack(mediaTrack" in backend


def test_preview_uses_the_published_track_without_cropping() -> None:
    backend = _BACKEND.read_text(encoding="utf-8")
    session = _SESSION.read_text(encoding="utf-8")
    app = _APP.read_text(encoding="utf-8")
    page = _PAGE.read_text(encoding="utf-8")

    assert "get cameraTrack()" in backend
    assert "get cameraTrack()" in session
    assert "model.session?.cameraTrack" in app
    assert "new MediaStream([track])" in app
    assert "navigator.mediaDevices.getUserMedia" not in app
    assert "object-fit: contain" in page
    assert "object-fit: cover" not in page
