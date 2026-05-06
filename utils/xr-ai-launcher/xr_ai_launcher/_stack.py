# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stack launcher with per-process readiness files.

Design
------
Every launchable sub-project is self-describing: it exposes an entry-point
command and accepts ``--config <path>.yaml`` (auto-discovered) and
``--ready-file <path>`` (injected by the launcher).

The stack is declared as a sequence of ``Process`` or ``Parallel`` items:

* ``Process`` — started alone; the launcher waits for it to signal ready
  before moving on.
* ``Parallel([p1, p2, ...])`` — all processes in the group are started at
  once; the launcher waits for *every* member to signal ready before the
  next item in the sequence begins.

For each process the launcher:

  1. Resolves the project directory and YAML config from the sample root.
  2. Spawns ``uv run --project <dir> <command> --config <yaml> --ready-file <f>``.
  3. Waits for the process to create *<f>* (the ready file), printing a
     progress line every five seconds so slow starts remain visible.
  4. Once all processes are ready, monitors them: any exit triggers a
     graceful shutdown of the rest.

Each process is responsible for creating its own ready file at the moment it
is fully initialized and able to serve requests — after model warm-up, after
the IPC socket connects, after the HTTP server starts listening, etc.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

from ._credentials import load_credentials

_READY_INTERVAL = 5.0   # seconds between progress lines
_STOP_TIMEOUT   = 20.0  # seconds before SIGKILL during shutdown

# launcher/ stays stdlib-only per AGENTS.md, so this module uses
# ``logging.getLogger`` rather than loguru. The orchestrator's
# ``setup_logging()`` installs an InterceptHandler that routes these
# stdlib records into loguru, so output ends up in the same sinks as
# the rest of the stack.
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Process:
    """
    Declares one process in the stack.

    name        — label used in log output.
    project     — path to the uv project (relative to the sample root, or absolute).
    command     — entry-point script to run inside the project's venv.
    config      — path to the YAML config (relative to the sample root, or absolute).
                  Passed as ``--config <path>`` to the subprocess. Omit for processes
                  that take no config.
    gpu         — optional CUDA_VISIBLE_DEVICES value (e.g. ``"0"``, ``"0,1"``).
    launch_mode — controls spawn + shutdown behaviour:
    port        — optional service port, used to stop ``persist`` services.

      ``"own"``     (default) — launcher spawns this process and kills it on shutdown.
      ``"persist"`` — launcher spawns this process but leaves it running on shutdown.
                      Use for heavy model servers that should survive stack restarts
                      (e.g. vLLM containers).  Cleanup is the caller's responsibility.
      ``"reuse"``   — launcher does NOT spawn this process; it is assumed to be already
                      running (e.g. started by ``model-servers``).  The entry in the
                      process list documents the dependency; the launcher skips it
                      entirely and does not kill it on shutdown.
    """
    name:        str
    project:     str | Path
    command:     str
    config:      str | Path | None = None
    gpu:         str | None = None
    launch_mode: str = "own"
    port:        int | None = None


@dataclass(frozen=True)
class Parallel:
    """
    A group of processes that start simultaneously.

    The launcher spawns every member at once and waits for *all* of them to
    signal readiness before advancing to the next item in the stack sequence.
    If any member exits before signaling ready the launcher shuts everything
    down, just as it would for a serial process.

    Example::

        Parallel([
            Process("stt", "../../ai-services/stt-server", "stt_server"),
            Process("tts", "../../ai-services/tts/piper",  "piper_tts_server"),
        ])
    """
    processes: tuple[Process, ...]

    def __init__(self, processes: Sequence[Process]) -> None:
        object.__setattr__(self, "processes", tuple(processes))


# ── subprocess helpers ─────────────────────────────────────────────────────────

def _forward(stream, prefix: str) -> None:
    """Drain *stream* line-by-line, printing each with *prefix*."""
    for raw in stream:
        print(f"{prefix} {raw.decode(errors='replace').rstrip()}", flush=True)


def _spawn(proc: Process, base: Path, ready_file: Path) -> subprocess.Popen:
    project = (base / proc.project).resolve()

    if shutil.which("uv"):
        # --quiet drops "Installed/Uninstalled N package" pre-run chatter.
        cmd: list[str] = ["uv", "run", "--quiet", "--project", str(project), proc.command]
    else:
        cmd = [sys.executable, "-m", proc.command]

    if proc.config is not None:
        cmd += ["--config", str((base / proc.config).resolve())]
    cmd += ["--ready-file", str(ready_file)]

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    if proc.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = proc.gpu

    # start_new_session=True puts uv + its children (e.g. xr_media_hub) in a
    # new process group.  _shutdown then kills the whole group so grandchild
    # processes don't survive as orphans when uv exits without forwarding signals.
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    prefix = f"[{proc.name}]"
    for stream in (p.stdout, p.stderr):
        threading.Thread(target=_forward, args=(stream, prefix), daemon=True).start()
    return p


# ── readiness wait ─────────────────────────────────────────────────────────────

def _wait_ready(name: str, ready_file: Path, proc: subprocess.Popen) -> None:
    """Block until *ready_file* exists. Print a progress line every 5 s."""
    t0          = time.monotonic()
    last_report = -_READY_INTERVAL  # force first line immediately

    while True:
        elapsed = time.monotonic() - t0

        if ready_file.exists():
            log.info("[%s] ready (%.0fs)", name, elapsed)
            return

        rc = proc.poll()
        if rc is not None:
            log.error("[%s] exited (rc=%s) before signaling ready", name, rc)
            raise SystemExit(1)

        if elapsed - last_report >= _READY_INTERVAL:
            log.debug("[%s] waiting... (%.0fs)", name, elapsed)
            last_report = elapsed

        time.sleep(0.5)


_ReadyEntry = tuple[str, Path, subprocess.Popen]

def _wait_ready_parallel(group: list[_ReadyEntry]) -> None:
    """Wait for all processes in *group* concurrently; raise if any fails."""
    failed: list[SystemExit] = []
    lock = threading.Lock()

    def _one(name: str, ready_file: Path, proc: subprocess.Popen) -> None:
        try:
            _wait_ready(name, ready_file, proc)
        except SystemExit as exc:
            with lock:
                failed.append(exc)

    threads = [
        threading.Thread(target=_one, args=entry, daemon=True)
        for entry in group
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failed:
        raise failed[0]


# ── monitor + shutdown ─────────────────────────────────────────────────────────

def _monitor(procs: dict[str, subprocess.Popen]) -> None:
    """Block until any process exits or SIGINT / SIGTERM is received."""
    stop = threading.Event()

    orig_int  = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)

    def _on_signal(sig, _frame):
        stop.set()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop.is_set():
            for name, p in procs.items():
                if p.poll() is not None:
                    log.warning("[%s] exited (rc=%s)", name, p.returncode)
                    return
            time.sleep(1.0)
    finally:
        signal.signal(signal.SIGINT,  orig_int)
        signal.signal(signal.SIGTERM, orig_term)


def _killpg(p: subprocess.Popen, sig: int) -> None:
    """Send *sig* to the process group of *p* (covers uv + its children)."""
    try:
        os.killpg(os.getpgid(p.pid), sig)
    except (ProcessLookupError, OSError):
        pass


def _shutdown(
    procs: dict[str, subprocess.Popen],
    no_kill: set[str] | None = None,
) -> None:
    """Terminate all running processes; escalate to SIGKILL after the timeout.

    Processes whose names are in *no_kill* (launch_mode "persist" or "reuse")
    are left running — their underlying service outlives this launcher.
    """
    skip = no_kill or set()
    for name, p in procs.items():
        if name in skip:
            log.info("[%s] keeping alive (launch_mode=persist)", name)
            continue
        if p.poll() is None:
            log.info("[%s] stopping…", name)
            _killpg(p, signal.SIGTERM)

    deadline = time.monotonic() + _STOP_TIMEOUT
    for name, p in procs.items():
        if name in skip:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if p.poll() is None:
            try:
                p.wait(timeout=max(0.1, remaining))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                log.warning("[%s] force-killing", name)
                _killpg(p, signal.SIGKILL)
                try:
                    p.wait(timeout=5.0)
                except Exception as exc:
                    log.warning(
                        "[%s] wait() failed after SIGKILL: %s — process may be zombie",
                        name, exc,
                    )


# ── public API ─────────────────────────────────────────────────────────────────

def run_stack(
    processes: Sequence[Union[Process, Parallel]],
    base: Path,
    *,
    exit_after_ready: bool = False,
) -> None:
    """
    Start *processes* in declaration order, waiting for each item to signal
    readiness before advancing to the next.

    Each item is either a single ``Process`` (started and awaited alone) or a
    ``Parallel`` group (all members started at once; the launcher waits for
    *every* member before moving on).

    Each process receives ``--ready-file <path>``.  When fully initialized
    it must ``Path(ready_file).touch()`` to signal the launcher.  Progress
    is printed every five seconds so model-loading or CloudXR start-up
    remains visible.

    After all processes are ready the launcher monitors them: if any exits,
    all others are terminated and the launcher exits.  Pass
    ``exit_after_ready=True`` to return immediately once everything is ready
    instead — useful for launchers whose processes are all ``launch_mode="persist"``
    and should outlive the orchestrator (e.g. ``model-servers``).

    *base* is the sample root — all relative paths in ``Process.project``
    and ``Process.config`` are resolved against it::

        _BASE = Path(__file__).resolve().parent

        PROCESSES = [
            Process("hub",    "../../server-runtime", "xr_media_hub",
                    config="yaml/xr_media_hub.yaml"),
            Parallel([
                Process("stt", "../../ai-services/stt-server", "stt_server"),
                Process("tts", "../../ai-services/tts/piper",  "piper_tts_server"),
            ]),
            Process("worker", "worker", "my_worker",
                    config="yaml/my_worker.yaml"),
        ]

        def run() -> None:
            run_stack(PROCESSES, _BASE)
    """
    load_credentials()

    # "persist" and "reuse" processes are left running on shutdown.
    # "reuse" processes are not spawned at all — assumed already running.
    _no_kill: set[str] = {
        p.name
        for item in processes
        for p in (item.processes if isinstance(item, Parallel) else [item])
        if p.launch_mode in ("persist", "reuse")
    }

    launched: dict[str, subprocess.Popen] = {}

    with tempfile.TemporaryDirectory(prefix="xr-ai-") as _tmpdir:
        tmpdir = Path(_tmpdir)
        try:
            for item in processes:
                if isinstance(item, Parallel):
                    to_spawn = [p for p in item.processes if p.launch_mode != "reuse"]
                    if not to_spawn:
                        continue
                    group: list[_ReadyEntry] = []
                    for proc in to_spawn:
                        ready_file = tmpdir / f"{proc.name}.ready"
                        launched[proc.name] = _spawn(proc, base, ready_file)
                        group.append((proc.name, ready_file, launched[proc.name]))
                    print(f"  [parallel] starting: {', '.join(p.name for p in to_spawn)}",
                          flush=True)
                    _wait_ready_parallel(group)
                else:
                    if item.launch_mode == "reuse":
                        continue
                    ready_file = tmpdir / f"{item.name}.ready"
                    launched[item.name] = _spawn(item, base, ready_file)
                    _wait_ready(item.name, ready_file, launched[item.name])

            log.info("All processes ready.")
            if exit_after_ready:
                return
            _monitor(launched)

        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            _shutdown(launched, _no_kill)
