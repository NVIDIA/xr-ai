# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic subprocess context manager with per-process log file routing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

_STOP_TIMEOUT = 20.0  # seconds before SIGKILL (docker compose down can take ~10 s)

# Lines emitted by a custom ``lovr.log`` callback (see render-mcp's main.lua)
# carry an authoritative severity from LOVR's official level vocabulary
# (debug | info | warn | error per https://lovr.org/docs/lovr.log).
_LOVR_LOG_PREFIX = re.compile(r"^LOVR_LOG\t(?P<level>\w+)\t(?P<tag>[^\t]*)\t(?P<msg>.*)$")

# Fallback for native C-level bytes that bypass lovr.log entirely (OpenXR
# loader, Vulkan loader, AppImage extraction). LOVR-spec keywords plus a
# couple of native conventions, matched case-insensitively. Underscore is a
# word character in Python regex, so this won't false-match inside identifiers
# like "no_validation_error_in_create_ref_space".
_LOVR_TERMINAL_RE = re.compile(
    r"\b(error|warn|warning|fatal|panic)\b",
    re.IGNORECASE,
)


def _ts() -> str:
    """``HH:MM:SS.mmm`` matching the loguru format in xr_ai_logging.setup_logging."""
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now - int(now)) * 1000):03d}"


def _open_log_file(name: str) -> IO[str] | None:
    """Open ``<XR_AI_LOG_ROOT>/log_<ns>_<ts>/<name>.log`` for append.

    Returns ``None`` when the per-run env vars are not stamped (i.e. the
    parent never called :func:`xr_ai_logging.setup_logging`), so the caller
    can fall back to ``print`` and not silently drop the subprocess output.
    """
    root = os.environ.get("XR_AI_LOG_ROOT")
    ns   = os.environ.get("XR_AI_LOG_NAMESPACE")
    ts   = os.environ.get("XR_AI_LOG_TIMESTAMP")
    if not (root and ns and ts):
        return None
    log_dir = Path(root) / f"log_{ns}_{ts}"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return open(log_dir / f"{name}.log", "a", buffering=1, encoding="utf-8")
    except OSError:
        return None


async def _forward(stream: asyncio.StreamReader, prefix: str, sink: IO[str] | None) -> None:
    """Drain *stream* line-by-line.

    Routing per line:

    * ``LOVR_LOG`` markers are parsed for authoritative severity. Records at
      ``warn`` or ``error`` are mirrored to the parent's stdout (visible in
      terminal); everything is written to *sink* as ``[level tag] message``.
    * Other lines are native C-level bytes (OpenXR loader, Vulkan, AppImage
      extraction). Mirrored to the parent's stdout iff the line contains
      ``error|warn|warning|fatal|panic`` (LOVR's level vocabulary plus common
      C conventions, case-insensitive); always written verbatim to *sink*.
    * When *sink* is ``None`` (env vars unset), fall back to today's
      unconditional prefixed ``print`` so output isn't silently dropped.
    """
    while True:
        line = await stream.readline()
        if not line:
            break
        decoded = line.decode(errors='replace').rstrip()

        if sink is None:
            print(f"{prefix} {decoded}", flush=True)
            continue

        match = _LOVR_LOG_PREFIX.match(decoded)
        if match:
            level = match["level"]
            tag   = match["tag"]
            msg   = match["msg"]
            file_line = f"[{level} {tag}] {msg}" if tag else f"[{level}] {msg}"
            try:
                sink.write(f"{_ts()} {file_line}\n")
            except (OSError, ValueError):
                # File closed mid-shutdown — fall back so we don't lose the line.
                print(f"{prefix} {file_line}", flush=True)
                continue
            if level in ("warn", "error"):
                print(f"{prefix} {file_line}", flush=True)
        else:
            try:
                sink.write(f"{_ts()} {decoded}\n")
            except (OSError, ValueError):
                print(f"{prefix} {decoded}", flush=True)
                continue
            if _LOVR_TERMINAL_RE.search(decoded):
                print(f"{prefix} {decoded}", flush=True)


@asynccontextmanager
async def ManagedProcess(name: str, cmd: list[str], cwd: Path | None = None,
                         env: dict[str, str] | None = None):
    """
    Run *cmd* as a subprocess and terminate it cleanly when the context exits.

    Output routing: when the parent has called
    :func:`xr_ai_logging.setup_logging` (per-run env vars stamped), stdout
    and stderr are appended line-by-line to ``<log_dir>/<name>.log`` with
    ``HH:MM:SS.mmm`` timestamps. Lines emitted via a custom ``lovr.log``
    callback (``LOVR_LOG\\t<level>\\t<tag>\\t<msg>``) are parsed for
    authoritative severity; ``warn``/``error`` records are also mirrored to
    the parent's stdout so they reach the terminal. Native bytes that don't
    carry the marker fall back to a regex match on LOVR's level vocabulary
    (case-insensitive ``error|warn|warning|fatal|panic``) before being
    mirrored.

    When the env vars are unset (standalone use), output is not lost: lines
    fall back to prefixed ``[name] …`` prints on the parent's stdout.

    Sends SIGTERM on exit; escalates to SIGKILL after ``_STOP_TIMEOUT``
    seconds. *env*, if given, replaces the child's environment entirely;
    otherwise the child inherits the parent's.
    """
    log.debug("[%s] starting: %s", name, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    log.debug("[%s] pid=%d", name, proc.pid)

    sink = _open_log_file(name)
    if sink is not None:
        log.info("[%s] output → %s", name, sink.name)

    prefix = f"[{name}]"
    pipe_tasks = [
        asyncio.create_task(_forward(proc.stdout, prefix, sink), name=f"{name}-stdout"),
        asyncio.create_task(_forward(proc.stderr, prefix, sink), name=f"{name}-stderr"),
    ]

    try:
        yield proc
    finally:
        if proc.returncode is None:
            log.debug("[%s] stopping (pid=%d)…", name, proc.pid)
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
        if sink is not None:
            try:
                sink.close()
            except OSError:
                pass
        log.debug("[%s] stopped (returncode=%s)", name, proc.returncode)
