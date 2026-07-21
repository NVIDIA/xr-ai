# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the XR render demo's typed scene process."""

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_nat.functions._rpc import RPCServer

from .engine import (
    SceneDispatcher,
    _build_config,
    _find_bundled_libzmq,
    _load_raw,
)
from .service import SceneService

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "scene_service.yaml"


async def _serve(config_path: Path, ready_file: Path | None) -> None:
    config = _build_config(config_path, _load_raw(config_path))
    os.environ["RENDER_SCENE_SOCKET"] = config.scene_socket
    bundled = _find_bundled_libzmq()
    if bundled is not None:
        os.environ["RENDER_ZMQ_LIB"] = str(bundled)
        logger.info("xr-render-scene: LOVR will load libzmq from {}", bundled)
    else:
        logger.warning("xr-render-scene: no bundled libzmq found; LOVR will use the system copy")

    async with contextlib.AsyncExitStack() as stack:
        dispatcher = SceneDispatcher(config, stack)
        service = SceneService(dispatcher)
        server = RPCServer(config.endpoint, service.dispatch)
        logger.info(
            "xr-render-scene rpc={} scene_socket={}",
            config.endpoint,
            config.scene_socket,
        )
        try:
            await server.serve(ready=ready_file.touch if ready_file else None)
        finally:
            await service.close()
            dispatcher.close()
            logger.info(
                "xr-render-scene stopped render_drops={}",
                dispatcher.health_snapshot()["render_drops"],
            )


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("xr-render-scene")
    config_path = args.config or _DEFAULT_CONFIG
    if not config_path.exists():
        sys.exit(f"xr-render-scene: config file not found: {config_path}")
    asyncio.run(_serve(config_path, args.ready_file))


if __name__ == "__main__":
    run()
