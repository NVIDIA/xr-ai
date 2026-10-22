# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port-to-process inspection shared by persistent servers and their cleanup."""
from __future__ import annotations

import os
import re
import select
import signal
import subprocess
from pathlib import Path

_STOP_TIMEOUT_S = 20.0
_KILL_TIMEOUT_S = 5.0


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


def _wait_for_process_exit(pidfd: int, timeout_s: float) -> bool:
    # Uvicorn closes the listener before draining requests and native work.
    # A pidfd becomes readable only when that exact process exits.
    return bool(select.select([pidfd], [], [], timeout_s)[0])


def stop_marked_listener(port: int, prefix: str, label: str) -> bool:
    """Stop the listener on *port* only if it carries *prefix*'s ownership marker.

    Linux only. Signals address a process handle captured before verification and
    wait for that exact process to exit, so a recycled PID is never signaled.
    """
    pid, checked, listening = pid_on_port_checked(port)
    if not checked or (listening and pid is None):
        print(f"{label}: cannot inspect ownership of port {port}; not stopping")
        return False
    if not listening:
        print(f"{label}: no persistent server is running")
        return True
    assert pid is not None
    pidfd = None
    try:
        pidfd = os.pidfd_open(pid)
        if not has_ownership_marker(pid, port, prefix):
            print(f"{label}: listener on port {port} is not an owned server; not stopping")
            return False
        if pid_on_port_checked(port) != (pid, True, True):
            print(f"{label}: listener on port {port} changed during inspection; not stopping")
            return False
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        if _wait_for_process_exit(pidfd, _STOP_TIMEOUT_S):
            print(f"{label}: stopped server on port {port}")
            return True
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        if _wait_for_process_exit(pidfd, _KILL_TIMEOUT_S):
            print(f"{label}: force-stopped server on port {port}")
            return True
    except ProcessLookupError:
        return True
    except OSError as exc:
        print(f"{label}: cannot track the configured process: {exc}; not stopping")
        return False
    finally:
        if pidfd is not None:
            os.close(pidfd)
    print(f"{label}: server on port {port} is still running")
    return False
