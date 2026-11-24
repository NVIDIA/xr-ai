# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the local model servers required by conversation-example."""

from __future__ import annotations

import argparse
from pathlib import Path

from xr_ai_model_server_stack import run_model_stack

_BASE = Path(__file__).resolve().parent
_SERVICES = ("speaker-stt", "tts", "omni", "clef")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop this sample's configured model servers and exit.",
    )
    parser.add_argument(
        "--gpu-profile",
        metavar="NAME",
        help=("Select an Omni hardware profile; with --stop, honor its port overrides. "
              "Gemma instead selects a GPU by free memory."),
    )
    args = parser.parse_args()
    try:
        from _profile_stack import _run_selected_profile

        if not _run_selected_profile(_BASE, stop=args.stop, gpu_profile=args.gpu_profile):
            run_model_stack(_BASE, _SERVICES, stop=args.stop, gpu_profile=args.gpu_profile)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run()
