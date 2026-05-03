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
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ._credentials import load_credentials
from ._project import ProjectLauncher

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Process:
    """
    Declares one process in the stack.

    name    — label used in log output.
    project — path to the uv project (relative to the sample root, or absolute).
    command — entry-point script to run inside the project's venv.
    gpu     — optional CUDA_VISIBLE_DEVICES value (e.g. "0", "1", "0,1").
              Omit to inherit the parent's GPU visibility.

    Config convention: ``run_stack`` looks for ``<command>.yaml`` in the
    sample root and passes it as ``--config <abs-path>`` if it exists.
    Processes with no YAML start with no extra arguments.
    """
    name:    str
    project: str | Path
    command: str
    gpu:     str | None = None


def _config_args(command: str, base: Path) -> list[str]:
    cfg = base / f"{command}.yaml"
    return ["--config", str(cfg)] if cfg.exists() else []


@asynccontextmanager
async def StackLauncher(processes: Sequence[Process], base: Path):
    """
    Start *processes* concurrently, resolving paths and configs from *base*.

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
