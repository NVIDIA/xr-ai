# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the settings for saving and captioning a procedure demonstration.

Read capture-agent-config.yaml into CaptureConfig: the output directory,
sampling rate, and caption interval. Load the caption prompt from prompts/ and
apply an optional output-directory override supplied by the worker configuration.
Shared voice and frame settings remain in the worker's config.py.

Resolve output paths beside the capture YAML so moving shared worker settings
does not change where a separately configured capture capability saves evidence.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Output, sampling, and caption settings used by the capture backend."""

    artifacts_dir: Path
    caption_prompt: str
    capture_fps: float = 2.0
    caption_interval_s: float = 5.0


def load_capture_config(path: Path, *, output_root: Path | None = None) -> CaptureConfig:
    """Read capture YAML and its bundled prompt, applying an output override if supplied."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = {}
    for name in ("capture_fps", "caption_interval_s"):
        if name in data:
            value = float(data[name])
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
            settings[name] = value
    return CaptureConfig(
        artifacts_dir=(
            output_root if output_root is not None
            else path.resolve().parent / data.get("artifacts_dir", "../artifacts")
        ).resolve(),
        caption_prompt=(Path(__file__).parent / "prompts/caption.txt").read_text(encoding="utf-8"),
        **settings,
    )
