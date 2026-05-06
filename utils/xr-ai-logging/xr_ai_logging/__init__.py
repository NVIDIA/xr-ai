# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared loguru setup for the xr-ai monorepo.

Every process in a sample run (orchestrator, worker, hub, AI services,
MCP servers, cloudxr) calls :func:`setup_logging` once at startup. The
result is a single, unified logging stack:

* **stderr sink** — level controlled by ``XR_AI_VERBOSE`` (DEBUG when
  truthy, INFO otherwise). What the user sees in their terminal.
* **file sink** — always DEBUG. Path:
  ``/tmp/log_<namespace>_<YYYY-MM-DD_HH-MM-SS>/<name>.log``
  (tmpfs-backed on most distros, so logs are RAM-resident and clear on
  reboot). One subfolder per run holds every process's log file for that
  run; the ``log_`` prefix makes the run folders easy to spot in ``/tmp``
  and to clean up with one ``rm`` glob. Loguru ``retention="7 days"``
  auto-prunes.
* **stdlib bridge** — a :class:`logging.Handler` (``_InterceptHandler``)
  routes any record emitted via ``logging.getLogger(...)`` into loguru.
  This is how ``utils/xr-ai-launcher/`` (stdlib-only by contract) and
  ``agent-sdk/xr_ai_agent/`` (pyzmq+msgpack-only by contract) end up in
  the same file/stderr sinks even though they cannot import loguru.

Subprocess coordination
-----------------------
The orchestrator stamps three env vars into ``os.environ`` so subsequent
``uv run`` subprocesses inherit them and produce log files inside the
same per-run subfolder:

* ``XR_AI_LOG_NAMESPACE`` — sample/process group (e.g. ``xr-render-demo``).
* ``XR_AI_LOG_TIMESTAMP`` — single ``YYYY-MM-DD_HH-MM-SS`` stamp per run.
* ``XR_AI_LOG_ROOT`` — absolute path to the directory that holds the per-run
  ``log_<namespace>_<timestamp>/`` folder. Defaults to ``/tmp``; set
  explicitly (e.g. in CI or for a debug session) to redirect the whole stack
  to a different root with one variable.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from loguru import logger

__all__ = ["setup_logging"]

_DEFAULT_LOG_ROOT = Path("/tmp")

# Stdlib loggers that emit a lot of low-value INFO/DEBUG noise. Pinned to
# WARNING so they don't drown the rest of the stream even in verbose mode.
# ``mcp.server.lowlevel.server`` emits one INFO per inbound tool call; the
# worker already logs the user-visible side via ``tool call`` / ``tool result``.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "mcp.server.lowlevel.server",
)

_TRUTHY = {"1", "true", "debug", "yes", "on"}


class _InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: int | str = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk back through frames inside the stdlib logging module so
        # loguru reports the original caller, not logging internals.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


def _is_verbose() -> bool:
    return os.environ.get("XR_AI_VERBOSE", "").strip().lower() in _TRUTHY


def _resolve_log_root() -> Path:
    """Return the directory that holds the per-run ``log_<ns>_<ts>/`` folder.

    Defaults to ``/tmp`` (tmpfs on most distros, so logs are RAM-backed and
    clear on reboot). The env var is stamped on first call so subprocesses
    inherit the same path even if their cwd differs; explicit callers (CI,
    a debug session) can set ``XR_AI_LOG_ROOT`` up-front to redirect the
    whole stack to disk-backed storage.
    """
    cached = os.environ.get("XR_AI_LOG_ROOT")
    if cached:
        return Path(cached)
    os.environ["XR_AI_LOG_ROOT"] = str(_DEFAULT_LOG_ROOT)
    return _DEFAULT_LOG_ROOT


def setup_logging(name: str, *, namespace: str | None = None) -> Path:
    """Install loguru sinks for this process. Returns the file path.

    Args:
        name: Process identifier shown in the log filename and stderr
            format (e.g. ``"orchestrator"``, ``"worker"``, ``"hub"``,
            ``"vlm"``, ``"stt"``).
        namespace: Optional grouping for related processes; typically the
            sample name (e.g. ``"xr-render-demo"``). Falls back to the
            ``XR_AI_LOG_NAMESPACE`` env var, then to ``name``. Setting it
            from the orchestrator and propagating via env keeps all
            subprocess logs together under the same directory.
    """
    verbose = _is_verbose()

    stamp = os.environ.get("XR_AI_LOG_TIMESTAMP") or time.strftime(
        "%Y-%m-%d_%H-%M-%S",
    )
    os.environ["XR_AI_LOG_TIMESTAMP"] = stamp

    ns = namespace or os.environ.get("XR_AI_LOG_NAMESPACE") or name
    os.environ["XR_AI_LOG_NAMESPACE"] = ns

    log_dir = _resolve_log_root() / f"log_{ns}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan> {message}"
        ),
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_file,
        level="DEBUG",
        retention="7 days",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} {level: <8} "
            "{name}:{function}:{line} {message}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(
        "logging initialised  name={}  namespace={}  verbose={}  file={}",
        name, ns, verbose, log_file,
    )
    return log_file
