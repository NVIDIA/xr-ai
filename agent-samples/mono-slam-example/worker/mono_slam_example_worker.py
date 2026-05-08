# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
mono-slam-example worker — entry point.

Launched as a subprocess by ``uv run mono_slam_example`` (the orchestrator).
Do not run this directly.

Config (mono_slam_example_worker.yaml — auto-passed by the launcher)
---------------------------------------------------------------------
    weights_path:       dpvo.pth    path to DPVO checkpoint (required)
    fov_deg:            60.0        horizontal FOV for approximate intrinsics
    focal_length_px:    <float>     if set, overrides fov_deg (calibrated value)
    frame_stride:       3           sample every Nth FrameSignal per track
    publish_viz:        true        push poses to viz side channel
    dpvo_cfg_overrides: []          list of YACS key=value strings for DPVO config
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

from agent import MonoSlamAgent

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    setup_logging("worker", namespace="mono-slam-example")

    weights_path = cfg.get("weights_path", "dpvo.pth")
    ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    agent = MonoSlamAgent(
        ep,
        weights_path       =str(weights_path),
        fov_deg            =float(cfg.get("fov_deg",           60.0)),
        focal_length_px    =cfg.get("focal_length_px",          None),
        frame_stride       =int(  cfg.get("frame_stride",       3)),
        publish_viz        =bool( cfg.get("publish_viz",        True)),
        dpvo_cfg_overrides =list( cfg.get("dpvo_cfg_overrides", [])),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    if ready_file:
        ready_file.touch()

    logger.info(
        "mono-slam-example connecting  sub={}  push={}  stride={}  weights={}",
        _HUB_PUB, _HUB_PUSH, cfg.get("frame_stride", 3), weights_path,
    )
    try:
        await agent.run()
    finally:
        agent.shutdown()
    logger.info("mono-slam-example stopped")


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
