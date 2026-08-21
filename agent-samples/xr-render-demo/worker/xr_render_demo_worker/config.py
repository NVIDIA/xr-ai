# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker configuration."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    models_config: pathlib.Path
    voice_gate_yaml: pathlib.Path

    scene_endpoint: str
    openxr_endpoint: str
    video_memory_endpoint: str
    text_memory_dir: str

    silence_duration:  float
    min_speech:        float
    silero_threshold:  float

    # None = disabled (default). A positive value opts in to auto-cancel.
    idle_timeout_secs: float | None


def load_config(path: pathlib.Path | None) -> WorkerConfig:
    data = _read_yaml(path)
    idle_timeout = data.get("idle_timeout_secs")

    return WorkerConfig(
        models_config = _resolve(path, "models.json"),
        voice_gate_yaml = _resolve(path, str(data.get("voice_gate_yaml", "voice_gate.yaml"))),
        scene_endpoint = data.get("scene_endpoint", "tcp://127.0.0.1:8320"),
        openxr_endpoint = data.get("openxr_endpoint", "tcp://127.0.0.1:8330"),
        video_memory_endpoint = data.get("video_memory_endpoint", "tcp://127.0.0.1:8310"),
        text_memory_dir = data.get("text_memory_dir", "/dev/shm/xr-ai/text-memory"),
        silence_duration  = float(data.get("silence_duration",  0.8)),
        min_speech        = float(data.get("min_speech",        0.15)),
        silero_threshold  = float(data.get("silero_threshold",  0.5)),
        idle_timeout_secs = float(idle_timeout) if idle_timeout else None,
    )


def _read_yaml(path: pathlib.Path | None) -> dict:
    if path and path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve(config_path: pathlib.Path | None, raw: str) -> pathlib.Path:
    p = pathlib.Path(raw)
    if config_path is not None and not p.is_absolute():
        return config_path.parent / p
    return p
