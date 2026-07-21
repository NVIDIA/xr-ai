# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the OpenXR tracking service."""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_launcher import load_cloudxr_env
from xr_ai_logging import setup_logging
from xr_ai_nat.functions._rpc import RPCServer

from .service import OpenXRService

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "openxr_service.yaml"


@dataclass(frozen=True)
class Config:
    endpoint: str
    cloudxr_env_file: Path | None


def _load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    env_value = raw.get("cloudxr_env_file")
    env_file = Path(env_value).expanduser() if env_value else None
    if env_file is not None and not env_file.is_absolute():
        env_file = (path.resolve().parent / env_file).resolve()
    return Config(
        endpoint=str(raw.get("endpoint", "tcp://0.0.0.0:8330")),
        cloudxr_env_file=env_file,
    )


async def _serve(config: Config, ready_file: Path | None) -> None:
    if config.cloudxr_env_file is not None:
        if config.cloudxr_env_file.exists():
            load_cloudxr_env(config.cloudxr_env_file)
        else:
            logger.error("OpenXR environment file not found: {}", config.cloudxr_env_file)

    from .session import HardwarePoseSource

    source = HardwarePoseSource()
    server = RPCServer(config.endpoint, OpenXRService(source).dispatch)
    logger.info("openxr-service rpc={}", config.endpoint)
    try:
        await server.serve(ready=ready_file.touch if ready_file else None)
    finally:
        await asyncio.to_thread(source.close)


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("openxr-service")
    config_path = args.config or _DEFAULT_CONFIG
    if not config_path.exists():
        sys.exit(f"openxr-service: config file not found: {config_path}")
    asyncio.run(_serve(_load_config(config_path), args.ready_file))


if __name__ == "__main__":
    run()
