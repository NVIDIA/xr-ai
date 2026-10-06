# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the local Clef SystemOne inference service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
from loguru import logger
from xr_ai_logging import setup_logging

from ._config import load_config
from ._service import create_app


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    setup_logging("clef-server")
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    try:
        config = load_config(args.config)
        if _reuse_ready_server(config.host, config.port, config.model_name, config.model_revision):
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


def _reuse_ready_server(host: str, port: int, model: str, revision: str) -> bool:
    """Reuse an existing listener only when its ready identity matches exactly."""
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.25):
            pass
    except OSError:
        return False
    try:
        with urlopen(f"http://{probe_host}:{port}/health", timeout=1.0) as response:
            payload = response.read(16_384)
    except (OSError, URLError) as exc:
        raise RuntimeError(f"port {port} is occupied by a listener without a usable Clef health endpoint") from exc
    try:
        health = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"port {port} returned invalid health data") from exc
    if (
        not isinstance(health, dict)
        or health.get("status") != "ready"
        or health.get("model") != model
        or health.get("model_revision") != revision
    ):
        raise RuntimeError(f"port {port} is occupied by a server that does not match {model}@{revision}")
    return True


if __name__ == "__main__":
    run()
