# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the browser camera behavior tests with Node's built-in test runner."""

import subprocess
from pathlib import Path

_TEST = Path(__file__).parent / "javascript/web_camera_capture.test.mjs"


def test_web_camera_capture_behavior() -> None:
    result = subprocess.run(
        ["node", str(_TEST)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
