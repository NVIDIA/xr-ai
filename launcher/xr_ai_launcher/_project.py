# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ProjectLauncher — runs any uv project as a managed subprocess.
"""
from __future__ import annotations

import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from ._processes import ManagedProcess


@asynccontextmanager
async def ProjectLauncher(project: str | Path, command: str, *args: str, name: str | None = None):
    """
    Run *command* inside *project*'s isolated venv via ``uv run --project``.
    Falls back to ``sys.executable -m <command>`` if uv is not on PATH.

    *name* is the label used in log output (the prefix on every forwarded
    line). When omitted it defaults to the trailing component of *command*.

    Yields the underlying ``asyncio.subprocess.Process`` so callers can
    ``await proc.wait()`` or inspect ``proc.returncode``.

        async with ProjectLauncher("../../server-runtime", "xr_media_hub") as proc:
            await proc.wait()
    """
    project = Path(project).resolve()
    if name is None:
        name = command.rsplit(".", 1)[-1]
    if shutil.which("uv"):
        cmd = ["uv", "run", "--project", str(project), command, *args]
    else:
        cmd = [sys.executable, "-m", command, *args]
    async with ManagedProcess(name, cmd) as proc:
        yield proc
