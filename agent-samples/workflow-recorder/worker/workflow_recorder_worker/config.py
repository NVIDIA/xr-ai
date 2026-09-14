# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the workflow recorder worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    models_config: Path
    voice_gate_yaml: Path
    artifacts_dir: Path
    guides_dir: Path
    caption_prompt: str
    capture_fps: float
    caption_interval_s: float
    guide_scan_interval_s: float
    frame_max_age_s: float
    frame_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None


def _resolve(config_path: Path | None, raw: str) -> Path:
    path = Path(raw)
    if config_path is not None and not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _positive(data: dict[str, Any], name: str, default: float) -> float:
    value = float(data.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_config(path: Path | None) -> WorkerConfig:
    data: dict[str, Any] = {}
    if path is not None and path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"worker config must be a YAML mapping: {path}")
        data = loaded
    idle_timeout = float(data.get("idle_timeout_secs", 0))
    prompt_path = _PACKAGE / "prompts" / "caption.txt"
    return WorkerConfig(
        models_config=_resolve(path, str(data.get("models_config", "models.json"))),
        voice_gate_yaml=_resolve(path, str(data.get("voice_gate_yaml", "voice_gate.yaml"))),
        artifacts_dir=_resolve(path, str(data.get("artifacts_dir", "../artifacts"))),
        guides_dir=_resolve(path, str(data.get("guides_dir", "../guides"))),
        caption_prompt=str(data.get("caption_prompt") or prompt_path.read_text(encoding="utf-8")).strip(),
        capture_fps=_positive(data, "capture_fps", 2.0),
        caption_interval_s=_positive(data, "caption_interval_s", 5.0),
        guide_scan_interval_s=_positive(data, "guide_scan_interval_s", 2.0),
        frame_max_age_s=_positive(data, "frame_max_age_s", 2.0),
        frame_timeout_s=_positive(data, "frame_timeout_s", 3.0),
        silence_duration=_positive(data, "silence_duration", 0.6),
        min_speech=_positive(data, "min_speech", 0.15),
        silero_threshold=_positive(data, "silero_threshold", 0.4),
        idle_timeout_secs=idle_timeout if idle_timeout > 0 else None,
    )
