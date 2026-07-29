# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker configuration."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    # Path to the models.yaml file (resolved relative to cwd).
    models_yaml: str

    # Path to the voice gate YAML.  Resolved the same way as models_yaml —
    # bare basenames sit next to this worker's config YAML.
    voice_gate_yaml: str

    scene_endpoint: str
    openxr_endpoint: str
    video_memory_endpoint: str
    text_memory_dir: str

    # VAD (Silero, ONNX).
    silence_duration:  float
    min_speech:        float
    silero_threshold:  float   # Silero speech probability gate (0..1)

    # Idle-timeout auto-cancel (seconds). None = disabled (default): a quiet
    # session is never cancelled for inactivity. A positive value opts in.
    idle_timeout_secs: float | None


def load_config(path: pathlib.Path | None) -> WorkerConfig:
    data: dict = {}
    if path and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    # Resolve models_yaml relative to the config file's directory so the path
    # works regardless of where the worker process is launched from. The
    # default `"models.yaml"` is a bare basename — it sits next to this
    # worker's config YAML in `agent-samples/xr-render-demo/yaml/`. When the
    # launcher passes `--config`, `path.parent` is that yaml dir; when run
    # bare without `--config`, the relative path falls back to CWD.
    #
    # `model_backend: nim` selects the NIM overlay for hosted LLM/VLM services.
    backend = str(data.get("model_backend", "local")).lower()
    models_yaml_raw = (
        "models.nim.yaml" if backend == "nim"
        else data.get("models_yaml", "models.yaml")
    )
    models_yaml = _resolve_relative(models_yaml_raw, path)
    voice_gate_yaml = _resolve_relative(
        data.get("voice_gate_yaml", "voice_gate.yaml"), path,
    )

    return WorkerConfig(
        models_yaml = models_yaml,
        voice_gate_yaml = voice_gate_yaml,
        scene_endpoint = data.get("scene_endpoint", "tcp://127.0.0.1:8320"),
        openxr_endpoint = data.get("openxr_endpoint", "tcp://127.0.0.1:8330"),
        video_memory_endpoint = data.get("video_memory_endpoint", "tcp://127.0.0.1:8310"),
        text_memory_dir = data.get("text_memory_dir", "/dev/shm/xr-ai/text-memory"),
        silence_duration  = float(data.get("silence_duration",  0.8)),
        min_speech        = float(data.get("min_speech",        0.15)),
        silero_threshold  = float(data.get("silero_threshold",  0.5)),
        # 0 / unset → disabled (None); a positive value opts into idle cancel.
        idle_timeout_secs = (float(data["idle_timeout_secs"])
                             if data.get("idle_timeout_secs") else None),
    )


def _resolve_relative(raw: str, config_path: pathlib.Path | None) -> str:
    p = pathlib.Path(raw)
    if config_path and not p.is_absolute():
        return str(config_path.parent / p)
    return raw
