# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ProjectLauncher — runs any uv project as a managed subprocess.
"""
from __future__ import annotations

import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from ._processes import ManagedProcess


@asynccontextmanager
async def ProjectLauncher(
    project: str | Path,
    command: str,
    *args: str,
    name: str | None = None,
    gpu: str | None = None,
):
    """
    Run *command* inside *project*'s isolated venv via ``uv run --project``.
    Falls back to ``sys.executable -m <command>`` if uv is not on PATH.

    *name* is the label used in log output (the prefix on every forwarded
    line). When omitted it defaults to the trailing component of *command*.

    *gpu* pins the process to specific device(s) by setting
    ``CUDA_VISIBLE_DEVICES`` in the child environment.  Pass a device index
    string (``"0"``, ``"1"``, ``"0,1"``) or omit to inherit the parent's
    visibility.  This is the only GPU-selection mechanism; no YAML key is
    read.

    Yields the underlying ``asyncio.subprocess.Process`` so callers can
    ``await proc.wait()`` or inspect ``proc.returncode``.

        async with ProjectLauncher("../../server-runtime", "xr_media_hub") as proc:
            await proc.wait()
    """
    project = Path(project).resolve()
    if name is None:
        name = command.rsplit(".", 1)[-1]

    # Always build an explicit env dict so we can safely inject env vars.
    # Drop VIRTUAL_ENV to prevent uv from warning about an active-venv mismatch.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    if shutil.which("uv"):
        cmd = ["uv", "run", "--project", str(project), command, *args]
    else:
        cmd = [sys.executable, "-m", command, *args]

    async with ManagedProcess(name, cmd, env=env) as proc:
        yield proc
