# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the live-monitoring task guide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    models_config: Path
    task_directory: Path
    caption_prompt: str
    frame_max_age_s: float
    frame_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    voice_gate_yaml: Path
    idle_timeout_secs: float | None


def _resolve(config_path: Path | None, value: str) -> Path:
    path = Path(value)
    return config_path.parent / path if config_path and not path.is_absolute() else path


def load_config(path: Path | None) -> WorkerConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path and path.exists() else {}
    data = data or {}
    prompt_path = _resolve(
        path,
        str(data.get("caption_prompt_file", "../worker/visual_task_guide_worker/prompts/caption.txt")),
    )
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt or len(prompt.encode("utf-8")) > 8_192:
        raise ValueError("caption prompt must contain 1..8192 UTF-8 bytes")
    idle = data.get("idle_timeout_secs")
    return WorkerConfig(
        models_config=_resolve(path, str(data.get("models_config", "models.local.json"))),
        task_directory=_resolve(path, str(data.get("task_directory", "../tasks/hand-counting"))),
        caption_prompt=prompt,
        frame_max_age_s=float(data.get("frame_max_age_s", 2.0)),
        frame_timeout_s=float(data.get("frame_timeout_s", 3.0)),
        silence_duration=float(data.get("silence_duration", 0.8)),
        min_speech=float(data.get("min_speech", 0.25)),
        silero_threshold=float(data.get("silero_threshold", 0.5)),
        voice_gate_yaml=_resolve(path, str(data.get("voice_gate_yaml", "voice_gate.yaml"))),
        idle_timeout_secs=float(idle) if idle is not None else None,
    )


__all__ = ["WorkerConfig", "load_config"]
