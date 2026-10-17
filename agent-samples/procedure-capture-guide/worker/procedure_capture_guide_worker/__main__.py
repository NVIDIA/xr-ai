# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the procedure capture and guide application defined in app.py.

The sample's outer main.py starts this worker alongside the hub. This entry
point reads the worker configuration and readiness-file arguments, loads settings
through config.py, and runs app.run_app() in an asyncio event loop. Refer to
app.py for model and tool wiring, agent registration, and resource cleanup.
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
