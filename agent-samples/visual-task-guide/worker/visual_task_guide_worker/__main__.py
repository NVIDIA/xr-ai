# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point."""

import argparse
import asyncio
from pathlib import Path

from .app import run_app
from .config import load_config


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ready-file", type=Path)
    args, _ = parser.parse_known_args()
    asyncio.run(run_app(load_config(args.config), ready_file=args.ready_file))


if __name__ == "__main__":
    run()
