# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for transport-independent media capture."""
from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging

from ._service import CaptureService
from .config import DEFAULT_CONFIG_NAME, load_capture_config


async def main(*, config_path: Path, ready_file: Path | None = None) -> None:
    setup_logging("capture")
    config = load_capture_config(config_path)
    service = CaptureService(config)
    await service.start()
    if ready_file is not None:
        ready_file.touch()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    run_task = asyncio.create_task(service.run(), name="capture-service")
    stop_task = asyncio.create_task(stop.wait(), name="capture-stop")
    try:
        done, _pending = await asyncio.wait(
            (run_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if run_task in done:
            run_task.result()
    finally:
        run_task.cancel()
        stop_task.cancel()
        await asyncio.gather(run_task, stop_task, return_exceptions=True)
        logger.info("media capture shutting down")
        await service.stop()


def run() -> None:
    parser = argparse.ArgumentParser(description="Capture normalized XR-AI hub media")
    parser.add_argument("--config", type=Path, default=Path.cwd() / DEFAULT_CONFIG_NAME)
    parser.add_argument("--ready-file", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main(config_path=args.config, ready_file=args.ready_file))


if __name__ == "__main__":
    run()
