# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the local Clef SystemOne inference service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_vllm._docker import _has_clef_ownership, pid_on_port_checked

from ._config import identity, load_config
from ._service import create_app

_MANAGED_ENV = "XR_AI_CLEF_MANAGED"
_PORT_ENV = "XR_AI_CLEF_PORT"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        markers = {_MANAGED_ENV: "1", _PORT_ENV: str(config.port)}
        if any(os.environ.get(key) != value for key, value in markers.items()):
            # /proc/<pid>/environ records the exec-time environment. Re-exec so
            # ownership remains independently verifiable after the launcher exits.
            os.execvpe(
                sys.executable,
                [sys.executable, "-m", "clef_server", *sys.argv[1:]],
                os.environ | markers,
            )
        setup_logging("clef-server")
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        if _reuse_ready_server(config):
            if args.ready_file is not None:
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                args.ready_file.touch()
            logger.info(
                "Reusing ready Clef model {} at revision {} on port {}",
                config.model_name,
                config.model_revision,
                config.port,
            )
            return
        asyncio.run(_serve(config, args.ready_file))
    except Exception:
        logger.exception("Clef server failed to start or stopped unexpectedly")
        raise


async def _serve(config, ready_file: Path | None) -> None:
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config),
            host=config.host,
            port=config.port,
            log_config=None,
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Clef HTTP server stopped before becoming ready")
            await asyncio.sleep(0.05)
        if ready_file is not None:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.touch()
        await task
    finally:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        if not task.done():
            server.should_exit = True
            await task


def _reuse_ready_server(config) -> bool:
    """Reuse only an owned listener with the exact requested configuration."""
    pid, checked, listening = pid_on_port_checked(config.port)
    if not checked or (listening and pid is None):
        raise RuntimeError(f"cannot inspect ownership of port {config.port}")
    if not listening:
        return False
    assert pid is not None
    if not _has_clef_ownership(pid, config.port):
        raise RuntimeError(
            f"port {config.port} belongs to an unmanaged listener; stop it before launching"
        )
    probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(config.host, config.host)
    url_host = f"[{probe_host}]" if ":" in probe_host else probe_host
    try:
        with urlopen(f"http://{url_host}:{config.port}/health", timeout=1.0) as response:
            payload = response.read(16_384)
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"managed listener on port {config.port} has no usable Clef health endpoint"
        ) from exc
    try:
        health = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"port {config.port} returned invalid health data") from exc
    if health != identity(config):
        raise RuntimeError(
            f"port {config.port} serves a different Clef configuration"
        )
    if pid_on_port_checked(config.port) != (pid, True, True):
        raise RuntimeError(f"listener on port {config.port} changed during inspection")
    return True


if __name__ == "__main__":
    run()
