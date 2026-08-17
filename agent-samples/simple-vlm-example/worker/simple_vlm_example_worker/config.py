# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the simple VLM worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "system.txt"


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Resolved runtime settings for one simple VLM worker."""

    models_config: Path
    voice_gate_yaml: Path
    system_prompt: str
    frame_max_age_s: float
    frame_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None


def _resolve(config_path: Path | None, raw: str) -> Path:
    path = Path(raw)
    if config_path is not None and not path.is_absolute():
        return config_path.parent / path
    return path


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"worker config must be a YAML mapping: {path}")
    return data


def _load_system_prompt(data: dict[str, Any], config_path: Path | None) -> str:
    if data.get("system_prompt") is not None:
        return str(data["system_prompt"])
    prompt_name = data.get("system_prompt_file")
    prompt_path = (
        _resolve(config_path, str(prompt_name))
        if prompt_name is not None
        else _DEFAULT_PROMPT
    )
    return prompt_path.read_text(encoding="utf-8")


def load_config(path: Path | None) -> WorkerConfig:
    """Load worker YAML and resolve its shared deployment profile."""

    data = _read_config(path)
    models_name = str(data.get("models_config", "models.local.json"))
    idle_timeout = data.get("idle_timeout_secs")

    return WorkerConfig(
        models_config=_resolve(path, models_name),
        voice_gate_yaml=_resolve(
            path,
            str(data.get("voice_gate_yaml", "voice_gate.yaml")),
        ),
        system_prompt=_load_system_prompt(data, path),
        frame_max_age_s=float(data.get("frame_max_age_s", 2.0)),
        frame_timeout_s=float(data.get("frame_timeout_s", 5.0)),
        silence_duration=float(data.get("silence_duration", 0.4)),
        min_speech=float(data.get("min_speech", 0.1)),
        silero_threshold=float(data.get("silero_threshold", 0.5)),
        idle_timeout_secs=float(idle_timeout) if idle_timeout else None,
    )
