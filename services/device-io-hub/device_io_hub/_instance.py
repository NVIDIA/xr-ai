# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
One hub per machine.

The hub claims resources that are fixed for the whole machine: the IPC socket
paths under ``/tmp``, the LiveKit container name, and the LiveKit host ports.
Neither claim fails while another hub holds it. ZeroMQ replaces an existing
``ipc://`` socket file on bind, and LiveKit startup force-removes a container
of the same name so a run that crashed can be restarted. Without a guard, a
second hub silently takes over the first one's resources and breaks the sample
that is running on it.

``hold_instance_lock`` takes an exclusive lock before the hub claims anything.
The kernel releases the lock when the process exits, however it exits, so
holding it proves that no other hub is alive and that whatever the hub finds
in its place was left by a run that is gone.
"""
from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from device_io_hub._errors import StartupError

LOCK_PATH = Path("/tmp/xr_hub.lock")

_BANNER = "━" * 56


@contextmanager
def hold_instance_lock(path: Path | None = None) -> Iterator[None]:
    """Hold the machine-wide hub lock for the duration of the block.

    *path* defaults to :data:`LOCK_PATH`. Raises :class:`StartupError` naming
    the running hub when another process holds the lock.
    """
    path = path or LOCK_PATH
    fd = _open(path)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _already_running(_owner(path)) from None
        _record_owner(fd)
        yield
    finally:
        # Closing the descriptor releases the lock.
        os.close(fd)


def _open(path: Path) -> int:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return _open_existing(path)
    try:
        # umask must not leave a persistent machine-wide lock unreadable to
        # the next user. Change the created inode, never replace its pathname.
        os.fchmod(fd, 0o644)
    except OSError:
        os.close(fd)
        raise
    return fd


def _open_existing(path: Path) -> int:
    try:
        return os.open(path, os.O_RDWR)
    except PermissionError:
        # Another user created the file. A read-only descriptor can still take
        # the lock; only recording this process as the owner is skipped.
        return os.open(path, os.O_RDONLY)


def _record_owner(fd: int) -> None:
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        # The descriptor is read-only (see _open); the lock is what matters.
        pass


def _owner(path: Path) -> str:
    """Describe the process recorded in *path*, for the refusal banner."""
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return "unknown process"
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return f"pid {pid}"
    command = " ".join(arg.decode(errors="replace") for arg in args if arg)
    return f"pid {pid}: {command}" if command else f"pid {pid}"


def _already_running(owner: str) -> StartupError:
    lines = [
        "",
        _BANNER,
        "  Another DeviceIOHub is already running — refusing to start",
        _BANNER,
        f"  Running hub : {owner}",
        "",
        "  Only one hub can run on a machine: hubs share fixed IPC socket",
        "  paths, LiveKit ports, and the LiveKit container. Starting a",
        "  second one would take them over and break the running sample.",
        "",
        "  Stop the running sample (Ctrl-C in its terminal) and retry.",
        _BANNER,
    ]
    return StartupError("\n".join(lines))
