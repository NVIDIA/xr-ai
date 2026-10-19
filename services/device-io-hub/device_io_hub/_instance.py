# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
One hub per machine.

Every hub names its LiveKit container ``xr-ai-livekit-server``, and by default
hubs also share the IPC socket paths under ``/tmp`` and the LiveKit host
ports. Neither claim fails while another hub holds it. LiveKit startup
force-removes a container of the same name so a run that crashed can be
restarted, and ZeroMQ replaces an existing ``ipc://`` socket file on bind.
Without a guard, a second hub silently takes over the first one's resources
and breaks the sample that is running on it.

:func:`acquire_instance_lock` takes an exclusive lock before the hub claims
anything. The kernel releases the lock when the process exits, however it
exits, so holding it proves that no other hub is alive and that whatever the
hub finds in its place was left by a run that is gone. The lock covers hubs
that share this ``/tmp``; a hub in a container with a private ``/tmp`` and
the host Docker socket is not covered.
"""
from __future__ import annotations

import fcntl
import os
import pwd
from contextlib import ExitStack
from pathlib import Path

from device_io_hub._errors import StartupError

LOCK_PATH = Path("/tmp/xr_hub.lock")

_BANNER = "━" * 56


def acquire_instance_lock(path: Path | None = None) -> int:
    """Take the machine-wide hub lock and return the descriptor that holds it.

    *path* defaults to :data:`LOCK_PATH`. The lock is held until the
    descriptor is closed or the process exits. Raises :class:`StartupError`
    when another process holds the lock or the lock file cannot be opened.
    """
    path = path or LOCK_PATH
    try:
        fd = _open(path)
    except PermissionError as exc:
        raise _cannot_open(path, exc) from None
    with ExitStack() as on_failure:
        on_failure.callback(os.close, fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _already_running(path) from None
        on_failure.pop_all()
    return fd


def _open(path: Path) -> int:
    # The lock needs no write access, so every user can take it on a 0644
    # file. O_EXCL keeps creation separate from opening another user's file,
    # which fs.protected_regular refuses for O_CREAT in sticky /tmp.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return os.open(path, os.O_RDONLY)
    with ExitStack() as on_failure:
        on_failure.callback(os.close, fd)
        # umask must not leave the file unreadable to the next user.
        os.fchmod(fd, 0o644)
        on_failure.pop_all()
    return fd


def _already_running(path: Path) -> StartupError:
    lines = [
        "",
        _BANNER,
        "  Another DeviceIOHub is already running — refusing to start",
        _BANNER,
        "  Only one hub can run on a machine: hubs share the LiveKit",
        "  container and, by default, the IPC socket paths and LiveKit",
        "  ports. Starting a second one would take them over and break",
        "  the running sample.",
        "",
        "  Stop the running sample (Ctrl-C in its terminal) and retry.",
        f"  To find it: fuser -v {path}",
        _BANNER,
    ]
    return StartupError("\n".join(lines))


def _cannot_open(path: Path, error: PermissionError) -> StartupError:
    try:
        owner = pwd.getpwuid(path.stat().st_uid).pw_name
    except (OSError, KeyError):
        # ignored: the owner only adds detail to a banner that already names
        # the file and the error.
        owner = "unknown"
    lines = [
        "",
        _BANNER,
        "  Cannot open the DeviceIOHub lock file — refusing to start",
        _BANNER,
        f"  File  : {path} (owner: {owner})",
        f"  Error : {error.strerror}",
        "",
        "  Every user's hub must be able to read this file. Once no hub",
        f"  is running, remove it (sudo rm {path}) and retry.",
        _BANNER,
    ]
    return StartupError("\n".join(lines))
