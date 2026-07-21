# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the video-memory service."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_nat.functions._rpc import RPCServer

from .service import VideoMemoryService
from .store import ChunkStore

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "video_memory_service.yaml"


async def _serve(config: dict, ready_file: Path | None) -> None:
    run_dir = os.environ.get("XR_RUN_DIR")
    default_out = str(Path(run_dir) / "frames") if run_dir else "/tmp/xr_video_queries"
    recordings_dir = str(config.get("recordings_dir", ""))
    store = ChunkStore(Path(recordings_dir)) if recordings_dir else None
    service = VideoMemoryService(
        store=store,
        out_dir=Path(config.get("out_dir") or default_out),
        gpu_id=int(config.get("gpu_id", 0)),
    )
    address = str(config.get("endpoint", "tcp://0.0.0.0:8310"))
    server = RPCServer(address, service.dispatch)
    logger.info(
        "video-memory-service rpc={} recording_enabled={}",
        address,
        store is not None,
    )
    await server.serve(ready=ready_file.touch if ready_file else None)


def _load_config(config_path: Path | None) -> dict:
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        if config_path is not None:
            sys.exit(f"video-memory-service: config file not found: {path}")
        return {}
    return yaml.safe_load(path.read_text()) or {}


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("video-memory-service")
    config = _load_config(args.config)
    asyncio.run(_serve(config, args.ready_file))


if __name__ == "__main__":
    run()
