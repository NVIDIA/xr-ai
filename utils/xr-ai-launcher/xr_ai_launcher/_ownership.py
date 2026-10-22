# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port-to-process inspection shared by persistent servers and their cleanup."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def pid_on_port_checked(port: int) -> tuple[int | None, bool, bool]:
    """Return the listening PID, inspection status, and listener presence.

    Tries `ss` first (always present on modern Linux), falls back to `lsof`.
    A listener without a visible PID is still reported so callers fail closed
    instead of mistaking an uninspectable listener for an unused port.
    """
    try:
        out = subprocess.check_output(
            ["ss", "-tlnpH", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1)), True, True
        return None, True, bool(out.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out.splitlines()[0]), True, True
        return None, True, False
    except subprocess.CalledProcessError:
        return None, True, False
    except FileNotFoundError:
        return None, False, False


def has_ownership_marker(pid: int, port: int, prefix: str) -> bool:
    """Return whether *pid* was started with ``<prefix>_MANAGED=1`` and ``<prefix>_PORT=<port>``."""
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return f"{prefix}_MANAGED=1".encode() in entries and f"{prefix}_PORT={port}".encode() in entries
