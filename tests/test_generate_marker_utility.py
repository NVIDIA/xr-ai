# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone marker PNG generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2

_SCRIPT = (
    Path(__file__).parents[1]
    / "agent-sdk/xr-ai-tools/xr_ai_tools/utilities/generate_marker.py"
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_generate_qr_png(tmp_path: Path) -> None:
    output = tmp_path / "qr.png"

    result = _run("qr", "XR AI marker utility", "--size", "256", "--output", str(output))

    assert result.returncode == 0, result.stderr
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert image is not None and image.shape == (256, 256)
    value, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    assert value == "XR AI marker utility"


def test_generate_aruco_png(tmp_path: Path) -> None:
    output = tmp_path / "aruco.png"

    result = _run(
        "aruco",
        "42",
        "--dictionary",
        "DICT_6X6_250",
        "--size",
        "256",
        "--margin",
        "16",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    assert image is not None and image.shape == (256, 256)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    )
    _, identifiers, _ = detector.detectMarkers(image)
    assert identifiers is not None and identifiers.reshape(-1).tolist() == [42]


def test_rejects_invalid_marker_arguments(tmp_path: Path) -> None:
    output = tmp_path / "marker.jpg"

    result = _run("aruco", "50", "--output", str(output))

    assert result.returncode == 2
    assert "marker ID must be between 0 and 49" in result.stderr
