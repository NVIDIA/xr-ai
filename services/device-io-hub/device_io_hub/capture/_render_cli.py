# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for derived capture rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging

from .renderer import CaptureRenderer


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Render a captioned H.264/AAC MP4 from a raw capture bundle",
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--bitrate", type=int, default=6_000_000)
    parser.add_argument("--overlay-seconds", type=float, default=12.0)
    parser.add_argument("--overlay-lines", type=int, default=4)
    args = parser.parse_args()
    setup_logging("capture-renderer")
    output = CaptureRenderer(
        gpu_id=args.gpu_id,
        bitrate=args.bitrate,
        overlay_seconds=args.overlay_seconds,
        overlay_lines=args.overlay_lines,
    ).render(args.bundle)
    logger.info("capture rendering completed path={}", output)


if __name__ == "__main__":
    run()
