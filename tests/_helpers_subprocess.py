# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared subprocess-test helpers — port selection + HTTP health probe.

The server-under-subprocess tests (GPU LLM servers, Piper TTS, the LiveKit
integration suite) each picked a free port and polled an HTTP ``/health``
endpoint with near-identical code. Those primitives live here so the three
suites share one implementation.

Only the leaf primitives are shared. The async readiness pollers stay
per-suite: they differ by transport (``urllib`` health GET vs raw
``asyncio.open_connection``) and by what "ready" means for that server.
"""
from __future__ import annotations

import socket
import urllib.request


def port_is_free(port: int) -> bool:
    """Return True if *port* can be bound on 127.0.0.1 right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_free_port(preferred: int | None = None) -> int:
    """Return a free TCP port on 127.0.0.1.

    With *preferred* set, return it if it binds, otherwise fall through to a
    kernel-assigned ephemeral port. With no *preferred*, ask the kernel to pick
    one atomically (bind to 0).
    """
    if preferred is not None and port_is_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def health_ok(port: int, timeout: float = 2.0) -> bool:
    """True iff ``GET http://127.0.0.1:<port>/health`` returns HTTP 200."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout,
        ) as r:
            return r.status == 200
    # Connection refused / DNS / timeout while the server is still booting — caller retries.
    except Exception:
        return False
