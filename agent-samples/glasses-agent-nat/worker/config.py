# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""glasses-agent-nat worker configuration."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    # Services
    stt_server:       str   # STT server, port 8103
    tts_server:       str   # Piper TTS, port 8105
    llm_server:       str   # Llama-Nemotron-8B (quick-ack + classification), port 8106
    agent_llm_server: str   # Nemotron-3-Nano-30B (agentic tool-calling loop),  port 8107
    vlm_mcp:          str   # vlm-mcp base URL, e.g. http://localhost:8240
    video_mcp:        str   # video-mcp base URL, e.g. http://localhost:8210
    transcript_mcp:   str   # transcript-mcp base URL, e.g. http://localhost:8200
    nat_workflow_config: pathlib.Path
    models_yaml:         pathlib.Path  # logical-name → spec map for xr-ai-models clients

    # Background VLM observation loop
    vlm_interval_s:            float
    vlm_obs_max:               int
    condenser_interval_s:      float
    transcript_source:         str
    guidance_check_interval_s: float  # how often to check step completion during guidance

    # VAD
    silence_threshold: float
    silence_duration:  float
    min_speech:        float
    silero_threshold:  float
    vad_noise_mult:    float


def load_config(path: pathlib.Path | None) -> WorkerConfig:
    data: dict = {}
    if path and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    base_dir = path.parent if path else pathlib.Path(__file__).resolve().parent.parent
    workflow_config = pathlib.Path(
        data.get("nat_workflow_config", "yaml/glasses_agent_nat_workflow.yaml")
    )
    if not workflow_config.is_absolute():
        workflow_config = (base_dir / workflow_config).resolve()

    models_yaml = pathlib.Path(data.get("models_yaml", "yaml/models.yaml"))
    if not models_yaml.is_absolute():
        models_yaml = (base_dir / models_yaml).resolve()

    return WorkerConfig(
        stt_server          = data.get("stt_server",          "http://localhost:8103"),
        tts_server          = data.get("tts_server",          "http://localhost:8105"),
        llm_server          = data.get("llm_server",          "http://localhost:8106"),
        agent_llm_server    = data.get("agent_llm_server",    "http://localhost:8107"),
        vlm_mcp             = data.get("vlm_mcp",             "http://localhost:8240"),
        video_mcp           = data.get("video_mcp",           "http://localhost:8210"),
        transcript_mcp      = data.get("transcript_mcp",      "http://localhost:8200"),
        nat_workflow_config = workflow_config,
        models_yaml         = models_yaml,
        vlm_interval_s            = float(data.get("vlm_interval_s",            1.0)),
        vlm_obs_max               = int(data.get("vlm_obs_max",               240)),
        condenser_interval_s      = float(data.get("condenser_interval_s",   60.0)),
        transcript_source         = data.get("transcript_source",    "glasses-agent-nat"),
        guidance_check_interval_s = float(data.get("guidance_check_interval_s", 4.0)),
        silence_threshold   = float(data.get("silence_threshold", 0.005)),
        silence_duration    = float(data.get("silence_duration",  0.8)),
        min_speech          = float(data.get("min_speech",        0.15)),
        silero_threshold    = float(data.get("silero_threshold",  0.3)),
        vad_noise_mult      = float(data.get("vad_noise_mult",    4.0)),
    )
