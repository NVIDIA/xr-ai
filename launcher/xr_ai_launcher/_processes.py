# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic subprocess context manager with prefixed log forwarding.

Each managed subprocess's stdout/stderr is read line-by-line by ``_forward``
and tee'd to:

  * the launcher's terminal (with ``[name] `` prefix), via ``print(..., flush=True)``;
  * ``<XR_AI_LOG_DIR>/<name>.log`` (per-process file), if ``XR_AI_LOG_DIR`` is set;
  * ``<XR_AI_LOG_DIR>/combined.log`` (chronological merge across all processes),
    if ``XR_AI_LOG_DIR`` is set.

The PIPE tee captures everything that reaches subprocess stdout/stderr —
``print()`` calls, vLLM (post-execvp), NeMo native lines (when its custom
logger doesn't propagate to root), and C++ stderr writes (e.g.,
PyNvVideoCodec).  This is the only path into the file for those non-Python
emitters.

For Python ``logging`` records that DO go through root, the subprocess's
own FileHandler (configured by ``_setup_logging(cfg)``) writes them to
``<name>.log`` and ``combined.log`` directly with a below-threshold filter
so the records never duplicate the PIPE-tee output: stream-handled
(>= user_level) records reach the file via PIPE tee; everything else
reaches the file via the subprocess FileHandler.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

_STOP_TIMEOUT = 20.0  # seconds before SIGKILL (docker compose down can take ~10 s)


def _open_log(path: Path) -> IO[str] | None:
    """Open *path* for line-buffered append.  Returns ``None`` on OSError so
    callers can degrade gracefully when the log dir is read-only or missing."""
    try:
        return open(path, "a", encoding="utf-8", buffering=1)
    except OSError as exc:
        log.warning("could not open log file %s: %s", path, exc)
        return None


async def _forward(stream: asyncio.StreamReader,
                   prefix: str,
                   per_process_handle: IO[str] | None,
                   combined_handle: IO[str] | None) -> None:
    """Drain *stream* line-by-line; tee each line to terminal, per-process
    log file, and combined.log.  Each write flushes immediately so
    ``Ctrl+\\`` (SIGQUIT) loses at most the kernel-pipe buffer's worth of
    in-flight bytes."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = f"{prefix} {line.decode(errors='replace').rstrip()}"
        print(text, flush=True)
        if per_process_handle is not None:
            try:
                per_process_handle.write(text + "\n")
                per_process_handle.flush()
            except (OSError, ValueError):
                pass
        if combined_handle is not None:
            try:
                combined_handle.write(text + "\n")
                combined_handle.flush()
            except (OSError, ValueError):
                pass


@asynccontextmanager
async def ManagedProcess(name: str, cmd: list[str], cwd: Path | None = None,
                         env: dict[str, str] | None = None):
    """
    Run *cmd* as a subprocess, forward stdout/stderr prefixed with [name],
    and terminate it cleanly when the context exits.

    Sends SIGTERM on exit; escalates to SIGKILL after _STOP_TIMEOUT seconds.
    *env*, if given, replaces the child's environment entirely; otherwise
    the child inherits the parent's.

    When ``XR_AI_LOG_DIR`` is exported (set up by ``run_stack``), this
    function additionally opens ``<XR_AI_LOG_DIR>/<name>.log`` and
    ``<XR_AI_LOG_DIR>/combined.log`` in append mode, line-buffered, and
    passes the handles to the per-stream ``_forward`` tasks so every
    subprocess line lands in those files alongside the terminal.
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

    log_dir_str = os.environ.get("XR_AI_LOG_DIR")
    log_dir = Path(log_dir_str) if log_dir_str else None
    per_process_handle: IO[str] | None = None
    combined_handle:    IO[str] | None = None
    if log_dir is not None:
        per_process_handle = _open_log(log_dir / f"{name}.log")
        combined_handle    = _open_log(log_dir / "combined.log")

    prefix = f"[{name}]"
    pipe_tasks = [
        asyncio.create_task(
            _forward(proc.stdout, prefix, per_process_handle, combined_handle),
            name=f"{name}-stdout",
        ),
        asyncio.create_task(
            _forward(proc.stderr, prefix, per_process_handle, combined_handle),
            name=f"{name}-stderr",
        ),
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
        for h in (per_process_handle, combined_handle):
            if h is not None:
                try:
                    h.flush()
                    h.close()
                except (OSError, ValueError):
                    pass
        log.info("[%s] stopped (returncode=%s)", name, proc.returncode)
