# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker configuration with paths resolved beside its YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    models_config: Path
    workflow_config: Path
    applications_config: Path
    voice_gate_config: Path
    rag_endpoint: str
    frame_max_age_s: float
    frame_timeout_s: float
    vlm_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None


def _path(value: str, source: Path | None) -> Path:
    path = Path(value)
    return source.parent / path if source and not path.is_absolute() else path


def load_config(source: Path | None) -> WorkerConfig:
    raw: dict[str, Any] = {}
    if source is not None:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"worker config must be a mapping: {source}")
        raw = loaded
    idle = float(raw.get("idle_timeout_secs", 0))
    return WorkerConfig(
        models_config=_path(str(raw.get("models_config", "models.omni.json")), source),
        workflow_config=_path(str(raw.get("workflow_config", "workflow.yaml")), source),
        applications_config=_path(str(raw.get("applications_config", "applications.yaml")), source),
        voice_gate_config=_path(str(raw.get("voice_gate_config", "voice_gate.yaml")), source),
        rag_endpoint=str(raw.get("rag_endpoint", "tcp://127.0.0.1:8340")),
        frame_max_age_s=float(raw.get("frame_max_age_s", 3)),
        frame_timeout_s=float(raw.get("frame_timeout_s", 5)),
        vlm_timeout_s=float(raw.get("vlm_timeout_s", 15)),
        silence_duration=float(raw.get("silence_duration", 0.8)),
        min_speech=float(raw.get("min_speech", 0.15)),
        silero_threshold=float(raw.get("silero_threshold", 0.3)),
        idle_timeout_secs=idle or None,
    )


__all__ = ["WorkerConfig", "load_config"]
