# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sequential stack launcher with per-process readiness files.

Design
------
Every launchable sub-project is self-describing: it exposes an entry-point
command and accepts ``--config <path>.yaml`` (auto-discovered) and
``--ready-file <path>`` (injected by the launcher).

Processes start **one at a time** in declaration order.  For each process
the launcher:

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
from typing import Sequence

from ._credentials import load_credentials

_READY_INTERVAL = 5.0   # seconds between progress lines
_STOP_TIMEOUT   = 20.0  # seconds before SIGKILL during shutdown


@dataclass(frozen=True)
class Process:
    """
    Declares one process in the stack.

    name    — label used in log output.
    project — path to the uv project (relative to the sample root, or absolute).
    command — entry-point script to run inside the project's venv.
    config  — path to the YAML config (relative to the sample root, or absolute).
              Passed as ``--config <path>`` to the subprocess. Omit for processes
              that take no config.
    gpu     — optional CUDA_VISIBLE_DEVICES value (e.g. ``"0"``, ``"0,1"``).
    """
    name:    str
    project: str | Path
    command: str
    config:  str | Path | None = None
    gpu:     str | None = None


# ── subprocess helpers ─────────────────────────────────────────────────────────

def _forward(stream, prefix: str) -> None:
    """Drain *stream* line-by-line, printing each with *prefix*."""
    for raw in stream:
        print(f"{prefix} {raw.decode(errors='replace').rstrip()}", flush=True)


def _spawn(proc: Process, base: Path, ready_file: Path) -> subprocess.Popen:
    project = (base / proc.project).resolve()

    if shutil.which("uv"):
        cmd: list[str] = ["uv", "run", "--project", str(project), proc.command]
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
            print(f"  [{name}] ready ({elapsed:.0f}s)", flush=True)
            return

        rc = proc.poll()
        if rc is not None:
            print(f"  [{name}] exited (rc={rc}) before signaling ready", flush=True)
            raise SystemExit(1)

        if elapsed - last_report >= _READY_INTERVAL:
            print(f"  [{name}] waiting... ({elapsed:.0f}s)", flush=True)
            last_report = elapsed

        time.sleep(0.5)


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
                    print(f"  [{name}] exited (rc={p.returncode})", flush=True)
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


def _shutdown(procs: dict[str, subprocess.Popen]) -> None:
    """Terminate all running processes; escalate to SIGKILL after the timeout."""
    for name, p in procs.items():
        if p.poll() is None:
            print(f"  [{name}] stopping…", flush=True)
            _killpg(p, signal.SIGTERM)

    deadline = time.monotonic() + _STOP_TIMEOUT
    for name, p in procs.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if p.poll() is None:
            try:
                p.wait(timeout=max(0.1, remaining))
            except subprocess.TimeoutExpired:
                print(f"  [{name}] force-killing", flush=True)
                _killpg(p, signal.SIGKILL)
                try:
                    p.wait(timeout=5.0)
                except Exception:
                    pass


# ── public API ─────────────────────────────────────────────────────────────────

def run_stack(processes: Sequence[Process], base: Path) -> None:
    """
    Start *processes* sequentially, waiting for each to signal readiness
    before launching the next.

    Each process receives ``--ready-file <path>``.  When fully initialized
    it must ``Path(ready_file).touch()`` to signal the launcher.  Progress
    is printed every five seconds so model-loading or CloudXR start-up
    remains visible.

    After all processes are ready the launcher monitors them: if any exits,
    all others are terminated and the launcher exits.

    *base* is the sample root — all relative paths in ``Process.project``
    and ``Process.config`` are resolved against it::

        _BASE = Path(__file__).resolve().parent

        PROCESSES = [
            Process("hub",    "../../server-runtime", "xr_media_hub",
                    config="yaml/xr_media_hub.yaml"),
            Process("worker", "worker",               "my_worker",
                    config="yaml/my_worker.yaml"),
        ]

        def run() -> None:
            run_stack(PROCESSES, _BASE)
    """
    load_credentials()

    launched: dict[str, subprocess.Popen] = {}

    with tempfile.TemporaryDirectory(prefix="xr-ai-") as _tmpdir:
        tmpdir = Path(_tmpdir)
        try:
            for proc in processes:
                ready_file = tmpdir / f"{proc.name}.ready"
                launched[proc.name] = _spawn(proc, base, ready_file)
                _wait_ready(proc.name, ready_file, launched[proc.name])

            print("\n  All processes ready.\n", flush=True)
            _monitor(launched)

        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            _shutdown(launched)
