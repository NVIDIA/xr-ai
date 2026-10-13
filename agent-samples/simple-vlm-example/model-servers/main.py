# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the local model servers required by simple-vlm-example."""
from __future__ import annotations

import argparse
from pathlib import Path

from xr_ai_model_server_stack._stack import _run

_BASE = Path(__file__).resolve().parent
_SERVICES = ('stt', 'tts', 'vlm')


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop this sample's configured model servers and exit.",
    )
    parser.add_argument(
        "--gpu-profile", metavar="NAME",
        help="Select a checked-in GPU profile instead of automatic detection.",
    )
    args = parser.parse_args()
    try:
        _run(_BASE, _SERVICES, stop=args.stop, gpu_profile=args.gpu_profile)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run()
