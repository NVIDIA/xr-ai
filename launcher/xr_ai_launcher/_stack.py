# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StackLauncher — starts a sequence of processes, each in its own isolated venv.

Design
------
Every launchable sub-project is self-describing: it exposes an entry-point
command and owns a YAML config file named ``<command>.yaml``.

The orchestrator code declares WHICH projects to run (an architectural
decision); the launcher discovers each project's YAML automatically and
passes it as ``--config <path>``.  No separate launcher config file exists.

All processes start concurrently — no ordering is required or expressed.
Every process must tolerate its peers not being ready at startup.

Per-run log files
-----------------
``run_stack`` creates a per-run folder ``<repo_root>/logs/<run_id>/`` (or
``<XR_AI_LOG_DIR_ROOT>/<run_id>/`` if the env var is set) and exports
``XR_AI_LOG_DIR`` so subprocesses know where to write their FileHandler
output.  ``run_id = "<ISO-timestamp>_<sample>"``.  The launcher itself
writes ``launcher.log`` and ``combined.log`` in that folder; subprocesses
write ``<name>.log`` (managed by ``ManagedProcess`` in ``_processes.py``).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ._credentials import load_credentials
from ._project import ProjectLauncher

log = logging.getLogger(__name__)

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}
_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


# ── per-source level filters ─────────────────────────────────────────────────
#
# Above-threshold accepts records the StreamHandler should DISPLAY (terminal).
# Below-threshold accepts records the FileHandler should CAPTURE for postmortem
# (records the StreamHandler dropped).  Their union is every record; their
# intersection is empty — so file gets DEBUG-level capture without duplicates.

class _AboveThresholdFilter(logging.Filter):
    """Pass records at level >= per-source threshold (terminal display)."""

    def __init__(self, default: int, sources: dict[str, int] | None = None) -> None:
        super().__init__()
        self.default = default
        self.sources = sources or {}

    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, thr in self.sources.items():
            if record.name.startswith(prefix):
                return record.levelno >= thr
        return record.levelno >= self.default


class _BelowThresholdFilter(logging.Filter):
    """Pass records at level < per-source threshold (file capture for the
    levels the StreamHandler dropped)."""

    def __init__(self, default: int, sources: dict[str, int] | None = None) -> None:
        super().__init__()
        self.default = default
        self.sources = sources or {}

    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, thr in self.sources.items():
            if record.name.startswith(prefix):
                return record.levelno < thr
        return record.levelno < self.default


def _resolve_launcher_log_level() -> int:
    name = os.environ.get("XR_AI_LOG_LEVEL", "INFO").upper()
    if name not in _VALID_LEVELS:
        name = "INFO"
    return getattr(logging, name, logging.INFO)


def _setup_logging_launcher(log_dir: Path | None) -> None:
    """Configure the launcher's own root logger.  StreamHandler writes to
    stderr at the user level; FileHandlers (only when ``log_dir`` is set)
    capture DEBUG to ``launcher.log`` and ``combined.log`` for postmortem.

    Replaces any prior basicConfig output.  Idempotent — clears existing
    handlers before re-installing."""
    user_level = _resolve_launcher_log_level()
    formatter = logging.Formatter(_FORMAT)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)  # filter does the user-level work
    sh.setFormatter(formatter)
    sh.addFilter(_AboveThresholdFilter(user_level))
    root.addHandler(sh)

    if log_dir is not None:
        for path in (log_dir / "launcher.log", log_dir / "combined.log"):
            fh = logging.FileHandler(path, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            fh.addFilter(_BelowThresholdFilter(user_level))
            root.addHandler(fh)


# ── run-folder bookkeeping ───────────────────────────────────────────────────

def _find_repo_root(base: Path) -> Path:
    """Walk up from *base* looking for the xr-ai repo root.  A directory is
    the repo root if it contains BOTH ``AGENTS.md`` and ``DEPENDENCIES.md``.
    Falls back to ``base.parents[1]`` (the conventional layout
    ``<repo>/agent-samples/<sample>/``) if no marker is found."""
    for d in (base, *base.parents):
        if (d / "AGENTS.md").exists() and (d / "DEPENDENCIES.md").exists():
            return d
    return base.parents[1] if len(base.parents) >= 2 else base


def _compute_run_id(base: Path) -> str:
    """Per-run folder name: ``<ISO-timestamp>_<sample>``.  Uses ``T``/`-`
    separators so the name is filesystem-safe across platforms."""
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return f"{ts}_{base.name}"


def _create_log_dir(base: Path) -> Path | None:
    """Create the per-run log folder and return its path.  Honors the env
    override ``XR_AI_LOG_DIR_ROOT``.  Returns ``None`` if creation fails
    (logging falls back to terminal-only)."""
    try:
        root_dir = os.environ.get("XR_AI_LOG_DIR_ROOT")
        if root_dir:
            root_path = Path(root_dir).expanduser().resolve()
        else:
            root_path = _find_repo_root(base) / "logs"
        log_dir = root_path / _compute_run_id(base)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    except OSError:
        return None


def _write_manifest(log_dir: Path, base: Path, processes: Sequence[Process]) -> Path:
    """Write a starting manifest.  Caller updates it on exit with end-of-run
    fields via ``_finalize_manifest``."""
    path = log_dir / "manifest.json"
    payload = {
        "sample":     base.name,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host":       socket.gethostname(),
        "processes":  [{"name": p.name, "project": str(p.project), "command": p.command}
                       for p in processes],
        "exit_codes": {},
        "ended_at":   None,
    }
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError:
        pass
    return path


def _finalize_manifest(path: Path,
                       procs: dict[str, asyncio.subprocess.Process]) -> None:
    """Update the run's ``manifest.json`` with ``ended_at`` and per-process
    ``exit_codes``.  Best-effort; never raises."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    payload["ended_at"]   = datetime.now().astimezone().isoformat(timespec="seconds")
    payload["exit_codes"] = {name: proc.returncode for name, proc in procs.items()}
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError:
        pass


# ── SIGQUIT handler ──────────────────────────────────────────────────────────
#
# Python's default SIGQUIT (Ctrl+\\) action is "core dump + exit", losing any
# unflushed buffer.  We override with a handler that flushes Python's logging
# system (FileHandler.emit already flushes per-record, but logging.shutdown
# closes any cached handles) and exits cleanly.  ``signal.signal`` is used
# instead of ``loop.add_signal_handler`` so the handler runs even when the
# asyncio loop is wedged.

def _install_sigquit_handler() -> None:
    def _on_sigquit(signum: int, frame) -> None:  # pragma: no cover — signal path
        try:
            logging.shutdown()
        except Exception:
            pass
        os._exit(128 + signum)

    try:
        signal.signal(signal.SIGQUIT, _on_sigquit)
    except (AttributeError, ValueError):
        # SIGQUIT may not be available on Windows; ignore.
        pass


@dataclass(frozen=True)
class Process:
    """
    Declares one process in the stack.

    name           — label used in log output.
    project        — path to the uv project (relative to the sample root, or absolute).
    command        — entry-point script to run inside the project's venv.
    gpu            — optional CUDA_VISIBLE_DEVICES value (e.g. "0", "1", "0,1").
                     Omit to inherit the parent's GPU visibility.
    health_url     — optional readiness URL.  When set, the launcher polls
                     ``GET <health_url>`` after spawning this process and
                     **blocks the spawn loop** until it returns 200 OK (or
                     ``health_timeout`` elapses, or the process exits).
                     Used to serialize vLLM startup so the cold-start KV
                     profile pass for one model doesn't race with another's
                     weight load — a memory race that vLLM's
                     ``gpu_memory_utilization`` cannot disambiguate (see
                     AGENTS.md "Process model").  Non-gated peers continue
                     to start concurrently.
    health_timeout — seconds to wait for ``health_url`` before failing the
                     stack.  Defaults to 600 because vLLM cold-start with
                     CUDA-graph capture or FlashInfer MoE autotune can take
                     3-8 min on first run.

    Config convention: ``run_stack`` looks for ``<command>.yaml`` in the
    sample root and passes it as ``--config <abs-path>`` if it exists.
    Processes with no YAML start with no extra arguments.
    """
    name:           str
    project:        str | Path
    command:        str
    gpu:            str | None = None
    health_url:     str | None = None
    health_timeout: float       = 600.0


def _config_args(command: str, base: Path) -> list[str]:
    cfg = base / f"{command}.yaml"
    return ["--config", str(cfg)] if cfg.exists() else []


# ── readiness gate for sequential vLLM startup ───────────────────────────────
#
# vLLM's ``--gpu-memory-utilization`` is a fraction of TOTAL device memory and
# subtracts ALL non-this-process GPU usage from the budget at profile time.
# When two vLLMs profile concurrently, each one's measurement includes the
# other's still-loading weights — a moving target that depends on disk speed
# and is impossible to budget around (every "Available KV cache memory: −X
# GiB" failure is a symptom of this race).  Sequential startup turns the
# target into a stable one: each vLLM sees only previous peers' fully-loaded
# steady state.  The gate is opt-in via ``Process.health_url``.

_HEALTH_POLL_INTERVAL = 2.0  # seconds between probes
_HEALTH_PROBE_TIMEOUT = 2.0  # per-probe HTTP timeout


async def _wait_for_health(
    name: str,
    url: str,
    timeout: float,
    proc: asyncio.subprocess.Process,
) -> None:
    """Poll ``GET url`` until 200 OK, or *proc* exits early, or *timeout*
    seconds elapse.  Stdlib only (``urllib.request``).  Each probe runs in
    the default thread executor so the launcher's asyncio loop is never
    blocked.

    Raises ``RuntimeError`` if *proc* exits before becoming healthy and
    ``TimeoutError`` if no successful probe occurs within *timeout*.
    Either propagates out of ``StackLauncher`` to trigger fail-fast
    teardown of every already-spawned peer."""
    log.info("[%s] waiting for health: %s (timeout=%.0fs)", name, url, timeout)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def _probe() -> int | None:
        try:
            with urllib.request.urlopen(url, timeout=_HEALTH_PROBE_TIMEOUT) as resp:
                return resp.status
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return None

    while loop.time() < deadline:
        if proc.returncode is not None:
            raise RuntimeError(
                f"[{name}] exited rc={proc.returncode} before becoming healthy"
            )
        status = await loop.run_in_executor(None, _probe)
        if status is not None and 200 <= status < 300:
            log.info("[%s] healthy", name)
            return
        await asyncio.sleep(_HEALTH_POLL_INTERVAL)

    raise TimeoutError(
        f"[{name}] not healthy at {url} within {timeout:.0f}s"
    )


@asynccontextmanager
async def StackLauncher(processes: Sequence[Process], base: Path):
    """
    Start *processes* in declared order, resolving paths and configs from
    *base*.  Processes are spawned concurrently by default; if a process
    declares ``health_url``, the spawn loop **blocks** on that URL
    returning 200 OK before spawning the next process.

    Sequencing is opt-in per process — non-gated peers continue to start
    concurrently, with the gated ones interleaved at their list position.
    See ``_wait_for_health`` for the rationale.

    *base* is the sample root directory — where the YAML configs live and
    relative project paths are anchored.

    Yields ``{name: asyncio.subprocess.Process}`` for optional monitoring.
    """
    log.info("stack: starting %d process(es) from %s", len(processes), base)
    async with contextlib.AsyncExitStack() as stack:
        procs: dict[str, asyncio.subprocess.Process] = {}
        for p in processes:
            project = (base / p.project).resolve()
            extra   = _config_args(p.command, base)
            proc    = await stack.enter_async_context(
                ProjectLauncher(project, p.command, *extra, name=p.name, gpu=p.gpu)
            )
            procs[p.name] = proc
            if p.health_url:
                await _wait_for_health(p.name, p.health_url, p.health_timeout, proc)
        yield procs


async def run_stack(processes: Sequence[Process], base: Path) -> None:
    """
    Start the stack and run until a signal or any process exits, then
    terminate all remaining processes.

    *base* is the sample root — pass ``Path(__file__).resolve().parents[1]``
    from the orchestrator so the stack works regardless of CWD::

        _BASE = Path(__file__).resolve().parents[1]

        PROCESSES = [
            Process("hub",    "../../server-runtime", "xr_media_hub"),
            Process("worker", "worker",               "my_agent_worker"),
        ]

        def run() -> None:
            asyncio.run(run_stack(PROCESSES, _BASE))
    """
    log_dir = _create_log_dir(base)
    if log_dir is not None:
        os.environ["XR_AI_LOG_DIR"] = str(log_dir)
        os.environ["XR_AI_LOG_NAME"] = "launcher"

    _setup_logging_launcher(log_dir)
    _install_sigquit_handler()

    if log_dir is not None:
        log.info("stack: per-run log folder %s", log_dir)
        manifest_path = _write_manifest(log_dir, base, processes)
    else:
        manifest_path = None

    load_credentials()  # inject any saved tokens before spawning child processes
    async with StackLauncher(processes, base) as procs:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        task_to_name = {
            asyncio.create_task(p.wait()): name
            for name, p in procs.items()
        }

        async def _watch() -> None:
            done, pending = await asyncio.wait(
                task_to_name.keys(), return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            first      = next(iter(done))
            name, rc   = task_to_name[first], first.result()
            if not stop.is_set():
                log.info("stack: %r exited (rc=%s) — stopping", name, rc)
            stop.set()

        watcher = asyncio.create_task(_watch(), name="stack-watcher")
        try:
            await stop.wait()
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            if manifest_path is not None:
                _finalize_manifest(manifest_path, procs)
