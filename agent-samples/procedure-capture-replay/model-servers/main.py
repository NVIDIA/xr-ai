# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the speech, language, and vision model servers for the sample.

This entry point selects batch and speaker-conditioned STT, acknowledgement TTS,
an LLM, and the captioning VLM. Capture uses batch STT without speaker enrollment,
TTS, and the VLM. Replay uses speaker-conditioned STT, TTS, the LLM, and the VLM
for guidance and verification. The shared model-stack
launcher combines server defaults with this directory's GPU-profile YAML and
handles startup, readiness, and explicit shutdown independently of the worker.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from xr_ai_model_server_stack import run_model_stack

_BASE = Path(__file__).resolve().parent
_SERVICES = ("speaker-stt", "stt", "tts", "omni", "vlm")


def run() -> None:
    """Start the sample's model servers, or stop them when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop this sample's configured model servers and exit.",
    )
    parser.add_argument(
        "--gpu-profile", metavar="NAME",
        help="Select a startup hardware profile; with --stop, apply its port overrides without GPU detection.",
    )
    args = parser.parse_args()
    try:
        run_model_stack(_BASE, _SERVICES, stop=args.stop, gpu_profile=args.gpu_profile)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run()
