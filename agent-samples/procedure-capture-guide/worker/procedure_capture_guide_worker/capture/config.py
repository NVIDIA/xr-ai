# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load capture, captioning, and quality-feedback settings for a demonstration.

Read capture-agent-config.yaml into CaptureConfig: the output directory,
sampling rate, caption interval, and quality-warning policy. Load the caption prompt from prompts/ and
apply an optional output-directory override supplied by the worker configuration.
Shared voice and frame settings remain in the worker's config.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .quality import QualityConfig


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Evidence collection and quality-feedback policy for capture mode."""

    artifacts_dir: Path
    caption_prompt: str
    capture_fps: float = 2.0
    caption_interval_s: float = 5.0
    quality: QualityConfig = field(default_factory=QualityConfig)


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
        quality=QualityConfig.model_validate(data.get("quality", {})),
        **settings,
    )
