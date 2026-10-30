# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the browser camera behavior tests with Node's built-in test runner."""

import subprocess
from pathlib import Path

import pytest

_TEST = Path(__file__).parent / "javascript/web_camera_capture.test.mjs"


@pytest.mark.parametrize("test_file", [_TEST, _TEST.with_name("photo_capture.test.mjs")])
def test_web_camera_capture_behavior(test_file) -> None:
    result = subprocess.run(
        ["node", str(test_file)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
