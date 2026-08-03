# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed worker configuration for the tea-making guided workflow sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Resolved runtime settings for one guided workflow worker."""

    models_yaml: Path
    voice_gate_yaml: Path
    workflow_yaml: Path
    rag_endpoint: str
    answer_prompt: Path
    frame_max_age_s: float
    frame_timeout_s: float
    vlm_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"worker config must be a YAML mapping: {path}")
    return data


def _resolve(raw: str, config_path: Path | None) -> Path:
    path = Path(raw)
    if config_path is not None and not path.is_absolute():
        return config_path.parent / path
    return path


def load_config(path: Path | None) -> WorkerConfig:
    """Load worker YAML and resolve paths relative to that file."""

    data = _read_config(path)
    idle_timeout = data.get("idle_timeout_secs")
    return WorkerConfig(
        models_yaml=_resolve(str(data.get("models_yaml", "models.yaml")), path),
        voice_gate_yaml=_resolve(
            str(data.get("voice_gate_yaml", "voice_gate.yaml")),
            path,
        ),
        workflow_yaml=_resolve(str(data.get("workflow_yaml", "workflow.yaml")), path),
        rag_endpoint=str(data.get("rag_endpoint", "tcp://127.0.0.1:8340")),
        answer_prompt=_resolve(
            str(data.get("answer_prompt", "../worker/tea_making_worker/prompts/system.txt")),
            path,
        ),
        frame_max_age_s=float(data.get("frame_max_age_s", 5.0)),
        frame_timeout_s=float(data.get("frame_timeout_s", 5.0)),
        vlm_timeout_s=float(data.get("vlm_timeout_s", 15.0)),
        silence_duration=float(data.get("silence_duration", 0.8)),
        min_speech=float(data.get("min_speech", 0.15)),
        silero_threshold=float(data.get("silero_threshold", 0.3)),
        idle_timeout_secs=float(idle_timeout) if idle_timeout else None,
    )


__all__ = ["WorkerConfig", "load_config"]
