# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the persistent local Clef-Flash 9B SystemOne service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from xr_ai_launcher import Process, read_config_scalar, run_stack
from xr_ai_logging import setup_logging
from xr_ai_vllm._docker import _stop_clef_server

_BASE = Path(__file__).resolve().parent


_CONFIG = _BASE / "yaml" / "clef_server.yaml"


def _port() -> int:
    try:
        port = int(read_config_scalar(_CONFIG, "port"))
    except ValueError as exc:
        raise ValueError(f"{_CONFIG}: port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{_CONFIG}: port must be between 1 and 65535")
    return port


def _stop_clef(port: int) -> bool:
    success, _, message = _stop_clef_server(port)
    if message:
        print(f"clef-flash: {message}")
    return success


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop only the Clef service configured by yaml/clef_server.yaml.",
    )
    args = parser.parse_args()
    setup_logging("orchestrator", namespace="clef-flash")
    root_uv_config = _BASE.parents[1] / "uv.toml"
    if root_uv_config.is_file():
        os.environ.setdefault("UV_CONFIG_FILE", str(root_uv_config))
    port = _port()
    if args.stop:
        if not _stop_clef(port):
            raise SystemExit("clef-flash: failed to stop the configured Clef server")
        return
    process = Process(
        "clef-flash",
        "../../services/clef-server",
        "clef_server",
        config=_CONFIG,
        launch_mode="persist",
        port=port,
    )
    run_stack([process], _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
