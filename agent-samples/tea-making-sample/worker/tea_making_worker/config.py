# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the native tea-making worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE = Path(__file__).resolve().parent


class _PackagedPrompt(str):
    """Mark prompt text loaded from a package default rather than an override."""


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    models_config: Path
    workflow_config: Path
    voice_gate_yaml: Path
    artifacts_dir: Path
    rag_endpoint: str
    foreground_prompt: str
    change_watch_caption_prompt: str
    change_watch_event_prompt: str
    transcript_summary_prompt: str
    video_caption_prompt: str
    video_delta_prompt: str
    change_watch_default_instruction: str
    change_watch_interval_s: float
    transcript_summary_interval_s: float
    video_log_interval_s: float
    background_history_size: int
    frame_max_age_s: float
    frame_timeout_s: float
    vlm_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None
    web_events_host: str
    web_events_port: int
    web_events_max_events: int


def _resolve(config_path: Path | None, raw: str) -> Path:
    path = Path(raw)
    if config_path is not None and not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"worker config must be a YAML mapping: {path}")
    return data


def _prompt(data: dict[str, Any], config_path: Path | None, name: str) -> str:
    inline = data.get(name)
    if inline is not None:
        return str(inline).strip()
    configured = data.get(f"{name}_file")
    path = (
        _resolve(config_path, str(configured))
        if configured is not None
        else _PACKAGE / "prompts" / f"{name}.txt"
    )
    text = path.read_text(encoding="utf-8").strip()
    return _PackagedPrompt(text) if configured is None else text


def _positive(data: dict[str, Any], name: str, default: float) -> float:
    value = float(data.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_config(path: Path | None) -> WorkerConfig:
    """Load YAML and resolve every filesystem path relative to that file."""

    data = _read_config(path)
    history_size = int(data.get("background_history_size", 50))
    if history_size <= 0:
        raise ValueError("background_history_size must be positive")
    idle_timeout = float(data.get("idle_timeout_secs", 0.0))
    web_events_host = str(data.get("web_events_host", "127.0.0.1")).strip()
    if not web_events_host:
        raise ValueError("web_events_host must not be empty")
    web_events_port = int(data.get("web_events_port", 8092))
    if not 0 <= web_events_port <= 65_535:
        raise ValueError("web_events_port must be between 0 and 65535")
    web_events_max_events = int(data.get("web_events_max_events", 5_000))
    if web_events_max_events <= 0:
        raise ValueError("web_events_max_events must be positive")
    return WorkerConfig(
        models_config=_resolve(path, str(data.get("models_config", "models.local.json"))),
        workflow_config=_resolve(path, str(data.get("workflow_config", "workflow.yaml"))),
        voice_gate_yaml=_resolve(path, str(data.get("voice_gate_yaml", "voice_gate.yaml"))),
        artifacts_dir=_resolve(path, str(data.get("artifacts_dir", "../artifacts"))),
        rag_endpoint=str(data.get("rag_endpoint", "tcp://127.0.0.1:8340")),
        foreground_prompt=_prompt(data, path, "foreground_prompt"),
        change_watch_caption_prompt=_prompt(data, path, "change_watch_caption_prompt"),
        change_watch_event_prompt=_prompt(data, path, "change_watch_event_prompt"),
        transcript_summary_prompt=_prompt(data, path, "transcript_summary_prompt"),
        video_caption_prompt=_prompt(data, path, "video_caption_prompt"),
        video_delta_prompt=_prompt(data, path, "video_delta_prompt"),
        change_watch_default_instruction=str(
            data.get(
                "change_watch_default_instruction",
                "important changes involving people, objects, actions, or possible hazards",
            )
        ),
        change_watch_interval_s=_positive(data, "change_watch_interval_s", 2.0),
        transcript_summary_interval_s=_positive(data, "transcript_summary_interval_s", 120.0),
        video_log_interval_s=_positive(data, "video_log_interval_s", 2.0),
        background_history_size=history_size,
        frame_max_age_s=_positive(data, "frame_max_age_s", 3.0),
        frame_timeout_s=_positive(data, "frame_timeout_s", 5.0),
        vlm_timeout_s=_positive(data, "vlm_timeout_s", 15.0),
        silence_duration=_positive(data, "silence_duration", 0.8),
        min_speech=_positive(data, "min_speech", 0.15),
        silero_threshold=_positive(data, "silero_threshold", 0.3),
        idle_timeout_secs=idle_timeout if idle_timeout > 0 else None,
        web_events_host=web_events_host,
        web_events_port=web_events_port,
        web_events_max_events=web_events_max_events,
    )


__all__ = ["WorkerConfig", "load_config"]
