# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private to xr_ai_vllm: identify a persisted server without adopting it."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import _docker


def persistent_server_identity(label: str, port: int) -> str | None:
    """Return an opaque instance identity for a recognized local model server.

    Return ``None`` when the port has no server. Container IDs distinguish
    replacements; local process identities include the boot ID and start time
    so a stale PID cannot qualify a replacement for model reuse. This does not
    check health or configuration. Discovery never sends a signal.

    Raises:
        RuntimeError: Ownership cannot be established or inspected safely.
    """
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if not label:
        raise ValueError("label must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError("port must be an integer")
    if not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    container, checked = _docker.container_on_port_checked(port)
    if not checked:
        raise RuntimeError(f"cannot inspect model ownership on port {port}")
    if container:
        try:
            identity = subprocess.check_output(
                ["docker", "inspect", "--format", "{{.Id}}:{{.State.StartedAt}}", container],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"cannot identify container on port {port}") from exc
        return f"container:{identity}"
    pid, checked, listening = _docker.pid_on_port_checked(port)
    if not checked or (listening and pid is None):
        raise RuntimeError(f"cannot inspect listener on port {port}")
    if not listening:
        return None
    if not _docker.is_xr_ai_server_process(pid, label, port):
        raise RuntimeError(f"port {port} belongs to an unmanaged listener")
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        # comm may contain spaces or parentheses; fields after its final ')' start at 3.
        started = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError) as exc:
        raise RuntimeError(f"cannot identify listener on port {port}") from exc
    if _docker.pid_on_port_checked(port) != (pid, True, True):
        raise RuntimeError(f"listener changed on port {port}")
    return f"process:{boot}:{pid}:{started}"
