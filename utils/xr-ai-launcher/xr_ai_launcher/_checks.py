# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for bounded launcher checks."""
from __future__ import annotations

import shutil
import subprocess


def _row(name: str, ok: bool | None, detected: str, required: str,
         remediation: str) -> dict[str, object]:
    status = "skipped" if ok is None else "passed" if ok else "failed"
    return {"name": name, "status": status, "detected": detected,
            "required": required, "remediation": "" if ok else remediation}


def _run(command: list[str], timeout: float = 5) -> tuple[bool, str]:
    if shutil.which(command[0]) is None:
        return False, f"{command[0]} not found"
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:g}s"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return True, result.stdout.strip()
    output = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
    if len(output) > 1000:
        output = output[:997] + "..."
    return False, output
