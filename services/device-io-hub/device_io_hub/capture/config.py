# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the optional media-hub capture process."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_NAME = "media_capture.yaml"
_DEFAULT_OUT_DIR = "~/.local/share/xr-ai/captures"


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    hub_sub_addr: str = "ipc:///tmp/xr_hub_pub"
    hub_push_addr: str = "ipc:///tmp/xr_hub_in"
    out_dir: str = _DEFAULT_OUT_DIR
    sample_fps: float = 30.0
    bitrate: int = 6_000_000
    gpu_id: int = 0
    frame_queue_size: int = 2
    encoder_workers: int = 2
    audio_sample_rate: int = 48_000
    overlay_seconds: float = 12.0
    overlay_lines: int = 4
    max_total_bytes: int = 10 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_fps) or not 1 <= self.sample_fps <= 120:
            raise ValueError("sample_fps must be finite and between 1 and 120")
        if self.bitrate <= 0:
            raise ValueError("bitrate must be positive")
        if self.gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
        if not 1 <= self.frame_queue_size <= 120:
            raise ValueError("frame_queue_size must be between 1 and 120")
        if not 1 <= self.encoder_workers <= 16:
            raise ValueError("encoder_workers must be between 1 and 16")
        if not 8_000 <= self.audio_sample_rate <= 192_000:
            raise ValueError("audio_sample_rate must be between 8000 and 192000")
        if not math.isfinite(self.overlay_seconds) or self.overlay_seconds < 0:
            raise ValueError("overlay_seconds must be finite and non-negative")
        if not 1 <= self.overlay_lines <= 12:
            raise ValueError("overlay_lines must be between 1 and 12")
        if self.max_total_bytes < 0:
            raise ValueError("max_total_bytes must be non-negative")


def load_capture_config(path: Path) -> CaptureConfig:
    """Load one capture config and resolve its output path."""
    if not path.exists():
        raise FileNotFoundError(f"Capture config file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("capture config must be a YAML mapping")

    values = dict(raw)
    out_dir = Path(str(values.get("out_dir", _DEFAULT_OUT_DIR))).expanduser()
    if not out_dir.is_absolute():
        out_dir = (path.parent / out_dir).resolve()
    values["out_dir"] = str(out_dir)
    return CaptureConfig(**values)
