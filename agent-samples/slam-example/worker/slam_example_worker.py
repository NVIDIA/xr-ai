# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
slam-example worker — entry point.

Launched as a subprocess by ``uv run slam_example`` (the orchestrator).
Do not run this directly.

Protocol
--------
Client → agent  (LiveKit data channel):
    Topic ``imu``         — batched DeviceMotionEvent JSON.
    Topic ``camera_meta`` — once per startCamera, describes the camera.

Agent → client:
    Topic ``pose.update`` — JSON {source, state, translation_m,
                                  quaternion, frames_sent, ts_ns}.

Config (slam_example_worker.yaml — auto-passed by the launcher)
----------------------------------------------------------------
    slam_mcp_url:          http://localhost:8260/mcp   # SLAM backend
    slam_hz:               2.0
    slam_max_age_s:        1.0
    slam_track_max_edge:   320
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal

import yaml
from loguru import logger
from xr_ai_agent import ProcessorEndpoint
from xr_ai_logging import setup_logging

from agent       import SlamAgent
from slam_client import SlamClient, wait_for_slam_mcp

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    setup_logging("worker")

    slam_url = cfg.get("slam_mcp_url") or None
    if not slam_url:
        raise SystemExit(
            "slam_example_worker requires `slam_mcp_url` in its YAML "
            "(point at one of pose-mcp / kimera-mcp / droid-mcp).",
        )
    slam = SlamClient(slam_url)
    await wait_for_slam_mcp(slam)

    if ready_file:
        ready_file.touch()

    ep    = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    agent = SlamAgent(
        ep, slam,
        slam_hz             =float(cfg.get("slam_hz",             2.0)),
        slam_max_age_s      =float(cfg.get("slam_max_age_s",      1.0)),
        slam_scratch_dir    =pathlib.Path(cfg.get(
            "slam_scratch_dir", "/dev/shm/xr-ai/slam-in")),
        slam_track_max_edge =int(  cfg.get("slam_track_max_edge", 320)),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    logger.info("slam-example worker connecting  sub={}  push={}", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        await slam.close()
    logger.info("slam-example worker stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(main(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
