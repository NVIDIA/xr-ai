# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic subprocess context manager with prefixed log forwarding.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

log = logging.getLogger(__name__)

_STOP_TIMEOUT = 20.0  # seconds before SIGKILL (docker compose down can take ~10 s)


async def _forward(stream: asyncio.StreamReader, prefix: str) -> None:
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f"{prefix} {line.decode(errors='replace').rstrip()}", flush=True)


@asynccontextmanager
async def ManagedProcess(name: str, cmd: list[str], cwd: Path | None = None,
                         env: dict[str, str] | None = None):
    """
    Run *cmd* as a subprocess, forward stdout/stderr prefixed with [name],
    and terminate it cleanly when the context exits.

    Sends SIGTERM on exit; escalates to SIGKILL after _STOP_TIMEOUT seconds.
    *env*, if given, replaces the child's environment entirely; otherwise
    the child inherits the parent's.
    """
    log.info("[%s] starting: %s", name, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    log.info("[%s] pid=%d", name, proc.pid)

    prefix = f"[{name}]"
    pipe_tasks = [
        asyncio.create_task(_forward(proc.stdout, prefix), name=f"{name}-stdout"),
        asyncio.create_task(_forward(proc.stderr, prefix), name=f"{name}-stderr"),
    ]

    try:
        yield proc
    finally:
        if proc.returncode is None:
            log.info("[%s] stopping (pid=%d)…", name, proc.pid)
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_STOP_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("[%s] did not exit cleanly — killing", name)
                proc.kill()
                await proc.wait()
        for t in pipe_tasks:
            t.cancel()
        await asyncio.gather(*pipe_tasks, return_exceptions=True)
        log.info("[%s] stopped (returncode=%s)", name, proc.returncode)
