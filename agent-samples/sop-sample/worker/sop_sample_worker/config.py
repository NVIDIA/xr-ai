# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture-only worker configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    models_config: Path
    voice_gate_yaml: Path
    artifacts_dir: Path
    media_capture_dir: Path
    caption_prompt: str
    capture_fps: float = 2.0
    caption_interval_s: float = 5.0
    frame_max_age_s: float = 2.0
    frame_timeout_s: float = 3.0
    silence_duration: float = 0.6
    min_speech: float = 0.15
    silero_threshold: float = 0.4


def load_config(path: Path | None) -> WorkerConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "yaml/worker.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent
    media_path = base / data.get("media_capture_yaml", "media_capture.yaml")
    media = yaml.safe_load(media_path.read_text(encoding="utf-8"))
    if media.get("session_mode") != "explicit":
        raise ValueError("SOP capture requires media session_mode: explicit")
    media_dir = Path(media["out_dir"]).expanduser()
    settings = {}
    for name in (
        "capture_fps",
        "caption_interval_s",
        "frame_max_age_s",
        "frame_timeout_s",
        "silence_duration",
        "min_speech",
        "silero_threshold",
    ):
        value = float(data[name]) if name in data else None
        if value is not None:
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
            settings[name] = value
    return WorkerConfig(
        models_config=base / "models.json",
        voice_gate_yaml=base / "voice_gate.yaml",
        artifacts_dir=(base / data.get("artifacts_dir", "../artifacts")).resolve(),
        media_capture_dir=(media_path.parent / media_dir).resolve(),
        caption_prompt=(Path(__file__).parent / "prompts/caption.txt").read_text(encoding="utf-8"),
        **settings,
    )
