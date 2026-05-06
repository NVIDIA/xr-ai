# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
pip-installed vLLM backend.

Spawns ``vllm serve …`` from the wrapper's venv. Persistent mode puts vLLM in a
new session group so the launcher's killpg() does not reach it; non-persistent
mode shares the wrapper's session so SIGTERM propagates and vLLM exits with
the wrapper.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from . import _lifecycle

log = logging.getLogger(__name__)


def run(
    *,
    persistent: bool,
    log_prefix: str,
    vllm_argv: list[str],
    host: str,
    port: int,
    ready_file: Path | None,
) -> None:
    health_url = _lifecycle.health_url(host, port)

    if persistent and _lifecycle.health_ok(health_url):
        print(
            f"[{log_prefix}] vLLM already running on port {port} — reusing",
            flush=True,
        )
        if ready_file:
            ready_file.touch()
        _lifecycle.idle_until_stopped(health_url, log_prefix)
        return

    print(
        f"[{log_prefix}] Launching vLLM (pip)  http://{host}:{port}/v1",
        flush=True,
    )
    # start_new_session=True is what makes persistence work — vLLM is in its
    # own process group so the launcher's killpg() on the wrapper does not
    # reach it. Non-persistent wrappers stay in the wrapper's group so SIGTERM
    # propagates and vLLM exits with the wrapper.
    proc = subprocess.Popen(vllm_argv, start_new_session=persistent)

    _lifecycle.wait_until_healthy(
        health_url,
        is_alive=lambda: proc.poll() is None,
    )

    log.info("Ready  →  http://localhost:%d/v1", port)
    if ready_file:
        ready_file.touch()

    if persistent:
        _lifecycle.idle_until_stopped(health_url, log_prefix)
    else:
        rc = proc.wait()
        if rc != 0:
            sys.exit(rc)
