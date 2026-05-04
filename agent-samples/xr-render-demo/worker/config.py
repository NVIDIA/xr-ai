# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker configuration."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    # Services
    stt_server:       str
    tts_server:       str
    llm_server:       str   # quick-ack LLM (Llama-Nemotron, port 8106)
    agent_llm_server: str   # tool-calling LLM (Nemotron-3-Nano, port 8107)
    render_mcp:       str   # base URL, e.g. http://localhost:8220
    oxr_mcp:          str   # base URL, e.g. http://localhost:8230
    vlm_server:       str   # VLM inference server, e.g. http://localhost:8100
    vlm_mcp:          str   # base URL, e.g. http://localhost:8240
    video_mcp:        str   # base URL, e.g. http://localhost:8210

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

    return WorkerConfig(
        stt_server       = data.get("stt_server",        "http://localhost:8103"),
        tts_server       = data.get("tts_server",        "http://localhost:8105"),
        llm_server       = data.get("llm_server",        "http://localhost:8106"),
        agent_llm_server = data.get("agent_llm_server",  "http://localhost:8107"),
        render_mcp       = data.get("render_mcp_url",    "http://localhost:8220"),
        oxr_mcp          = data.get("oxr_mcp_url",       "http://localhost:8230"),
        vlm_server       = data.get("vlm_server",         "http://localhost:8100"),
        vlm_mcp          = data.get("vlm_mcp_url",       "http://localhost:8240"),
        video_mcp        = data.get("video_mcp_url",     "http://localhost:8210"),
        silence_threshold = float(data.get("silence_threshold", 0.005)),
        silence_duration  = float(data.get("silence_duration",  0.8)),
        min_speech        = float(data.get("min_speech",        0.15)),
        silero_threshold  = float(data.get("silero_threshold",  0.5)),
        vad_noise_mult    = float(data.get("vad_noise_mult",    4.0)),
    )
