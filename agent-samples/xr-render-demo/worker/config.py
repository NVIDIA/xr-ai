# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker configuration."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    # Path to the models.yaml file (resolved relative to the worker YAML).
    models_yaml: str

    # MCP server base URLs — not in scope for xr-ai-models.
    render_mcp: str   # base URL, e.g. http://localhost:8220
    oxr_mcp:    str   # base URL, e.g. http://localhost:8230
    vlm_mcp:    str   # base URL, e.g. http://localhost:8240
    video_mcp:  str   # base URL, e.g. http://localhost:8210

    # VAD
    silence_threshold: float
    silence_duration:  float
    min_speech:        float
    silero_threshold:  float   # Silero speech probability gate (0..1)
    vad_noise_mult:    float   # adaptive energy fallback multiplier


def load_config(path: pathlib.Path | None) -> WorkerConfig:
    data: dict = {}
    if path and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    # Resolve models_yaml relative to the config file's directory so the path
    # works regardless of where the worker process is launched from. Note:
    # when ``path`` is None (no --config), the default "yaml/models.yaml" is
    # interpreted relative to CWD — the launcher always passes an absolute
    # --config, so this only bites bare ``uv run xr_render_demo_worker`` runs
    # from outside the worker directory.
    models_yaml_raw = data.get("models_yaml", "yaml/models.yaml")
    if path and not pathlib.Path(models_yaml_raw).is_absolute():
        models_yaml = str(path.parent / models_yaml_raw)
    else:
        models_yaml = models_yaml_raw

    return WorkerConfig(
        models_yaml = models_yaml,
        render_mcp  = data.get("render_mcp_url",  "http://localhost:8220"),
        oxr_mcp     = data.get("oxr_mcp_url",     "http://localhost:8230"),
        vlm_mcp     = data.get("vlm_mcp_url",     "http://localhost:8240"),
        video_mcp   = data.get("video_mcp_url",   "http://localhost:8210"),
        silence_threshold = float(data.get("silence_threshold", 0.005)),
        silence_duration  = float(data.get("silence_duration",  0.8)),
        min_speech        = float(data.get("min_speech",        0.15)),
        silero_threshold  = float(data.get("silero_threshold",  0.5)),
        vad_noise_mult    = float(data.get("vad_noise_mult",    4.0)),
    )
