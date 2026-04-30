# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HubLauncher — starts xr_media_hub as a managed subprocess.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from ._processes import ManagedProcess
from ._project import ProjectLauncher

_CONFIG_NAME = "xr_media_hub.yaml"


def _find_config(start: Path) -> Path | None:
    """Walk upward from *start* looking for xr_media_hub.yaml."""
    for p in [start, *start.parents]:
        c = p / _CONFIG_NAME
        if c.exists():
            return c
    return None


def _find_server_runtime(start: Path) -> Path | None:
    """Walk upward from *start* looking for a server-runtime/ project dir."""
    for p in [start, *start.parents]:
        sr = p / "server-runtime"
        if (sr / "pyproject.toml").exists():
            return sr
    return None


@asynccontextmanager
async def HubLauncher(config: str | Path | None = None):
    """
    Start xr_media_hub in a subprocess and stop it when the context exits.

    Config discovery: walks upward from CWD for xr_media_hub.yaml.
    Pass config=<path> to override.

    The hub is launched via ``uv run --project <server-runtime>`` so it runs
    in its own isolated environment regardless of the caller's venv.
    Falls back to sys.executable if uv is not on PATH.

        async with HubLauncher():
            await my_agent.run()
    """
    if config is None:
        config = _find_config(Path.cwd())

    config_path = Path(config) if config else None
    start = config_path.parent if config_path else Path.cwd()
    server_runtime = _find_server_runtime(start)

    extra = ["--config", str(config)] if config else []

    if server_runtime:
        async with ProjectLauncher(server_runtime, "xr_media_hub", *extra):
            yield
    else:
        async with ManagedProcess("hub", [sys.executable, "-m", "xr_media_hub", *extra]):
            yield
