"""
HubLauncher — starts xr_media_hub as a managed subprocess.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from ._processes import ManagedProcess

_CONFIG_NAME = "xr_media_hub.yaml"


def _find_config(start: Path) -> Path | None:
    """Walk upward from *start* looking for xr_media_hub.yaml."""
    for p in [start, *start.parents]:
        c = p / _CONFIG_NAME
        if c.exists():
            return c
    return None


@asynccontextmanager
async def HubLauncher(config: str | Path | None = None):
    """
    Start xr_media_hub in a subprocess and stop it when the context exits.

    Config discovery: walks upward from CWD for xr_media_hub.yaml.
    Pass config=<path> to override.

    The hub's stdout/stderr is forwarded with a [hub] prefix so it appears
    inline with the calling process's output.

        async with HubLauncher():
            await my_agent.run()
    """
    if config is None:
        config = _find_config(Path.cwd())

    cmd = [sys.executable, "-m", "xr_media_hub"]
    if config:
        cmd += ["--config", str(config)]

    async with ManagedProcess("hub", cmd):
        yield
