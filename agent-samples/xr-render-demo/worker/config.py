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

    # MCP server base URLs — not in scope for xr-ai-models.
    render_mcp: str   # base URL, e.g. http://localhost:8220
    oxr_mcp:    str   # base URL, e.g. http://localhost:8230
    vlm_mcp:    str   # base URL, e.g. http://localhost:8240
    video_mcp:  str   # base URL, e.g. http://localhost:8210
    vec_mcp:    str   # base URL, e.g. http://localhost:8250

    # VAD (Silero, ONNX).
    silence_duration:  float
    min_speech:        float
    silero_threshold:  float   # Silero speech probability gate (0..1)


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
    # `model_backend: nim` selects the NIM overlay (hosted LLM/VLM); the
    # orchestrator reads the same key to point vlm-mcp at its NIM config.
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
        render_mcp  = data.get("render_mcp_url",  "http://localhost:8220"),
        oxr_mcp     = data.get("oxr_mcp_url",     "http://localhost:8230"),
        vlm_mcp     = data.get("vlm_mcp_url",     "http://localhost:8240"),
        video_mcp   = data.get("video_mcp_url",   "http://localhost:8210"),
        vec_mcp     = data.get("vec_mcp_url",     "http://localhost:8250"),
        silence_duration  = float(data.get("silence_duration",  0.8)),
        min_speech        = float(data.get("min_speech",        0.15)),
        silero_threshold  = float(data.get("silero_threshold",  0.5)),
    )


def _resolve_relative(raw: str, config_path: pathlib.Path | None) -> str:
    p = pathlib.Path(raw)
    if config_path and not p.is_absolute():
        return str(config_path.parent / p)
    return raw
