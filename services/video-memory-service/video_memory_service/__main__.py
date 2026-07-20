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
from xr_ai_agent import ProcessorEndpoint, Subscribe
from xr_ai_logging import setup_logging
from xr_ai_nat.functions._rpc import RPCServer

from .live import LiveFrameProvider
from .service import VideoMemoryService
from .store import ChunkStore

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "video_memory_service.yaml"
_DEFAULT_HUB_PUB = "ipc:///tmp/xr_hub_pub"
_DEFAULT_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def _serve(config: dict, ready_file: Path | None) -> None:
    run_dir = os.environ.get("XR_RUN_DIR")
    default_out = str(Path(run_dir) / "frames") if run_dir else "/tmp/xr_video_queries"
    recordings_dir = str(config.get("recordings_dir", ""))
    store = ChunkStore(Path(recordings_dir)) if recordings_dir else None
    endpoint = ProcessorEndpoint(
        sub_addr=str(config.get("hub_pub", _DEFAULT_HUB_PUB)),
        push_addr=str(config.get("hub_push", _DEFAULT_HUB_PUSH)),
        filter=Subscribe.VIDEO,
    )
    service = VideoMemoryService(
        provider=LiveFrameProvider(endpoint),
        store=store,
        out_dir=Path(config.get("out_dir") or default_out),
        gpu_id=int(config.get("gpu_id", 0)),
    )
    address = str(config.get("endpoint", "tcp://0.0.0.0:8310"))
    server = RPCServer(address, service.dispatch)
    endpoint_task = asyncio.create_task(endpoint.run(), name="video-memory-hub-ipc")
    logger.info(
        "video-memory-service rpc={} recording_enabled={}",
        address,
        store is not None,
    )
    try:
        await server.serve(ready=ready_file.touch if ready_file else None)
    finally:
        endpoint.stop()
        endpoint_task.cancel()
        try:
            await endpoint_task
        except asyncio.CancelledError:
            pass
        except Exception as error:
            logger.debug("video-memory hub task stopped with {!r}", error)
        endpoint.close()


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("video-memory-service")
    config_path = args.config or _DEFAULT_CONFIG
    if not config_path.exists():
        sys.exit(f"video-memory-service: config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    asyncio.run(_serve(config, args.ready_file))


if __name__ == "__main__":
    run()
