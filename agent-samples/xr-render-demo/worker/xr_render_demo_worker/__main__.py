# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the xr-render-demo worker."""

from __future__ import annotations

import argparse
import asyncio
import pathlib

from .app import run_app
from .config import load_config


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    cfg = load_config(ns.config)
    asyncio.run(run_app(cfg, ns.config, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
