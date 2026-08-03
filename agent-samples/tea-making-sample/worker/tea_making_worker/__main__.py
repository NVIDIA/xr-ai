# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse launcher arguments and run the tea-making worker."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .app import run_app
from .config import load_config


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()
    asyncio.run(run_app(load_config(args.config), ready_file=args.ready_file))


if __name__ == "__main__":
    run()
