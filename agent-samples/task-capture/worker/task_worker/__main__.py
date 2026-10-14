# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-process CLI for voice-controlled task capture.

The outer main.py launches the hub and this worker as separate processes. This
entry point accepts the launcher's configuration and readiness-file arguments,
then starts the application that records snapshots, captions, and narration
between spoken start and stop commands.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from .app import run_app
from .config import load_config


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)

    asyncio.run(
        run_app(
            load_config(args.config),
            ready_file=args.ready_file,
        )
    )


if __name__ == "__main__":
    run()
