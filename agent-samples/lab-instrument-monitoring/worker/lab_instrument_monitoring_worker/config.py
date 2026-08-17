# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the lab instrument monitoring worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .device_map import DeviceMap, load_device_map

_PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    models_config: Path
    voice_gate_yaml: Path
    device_map: DeviceMap
    artifacts_dir: Path
    foreground_prompt: str
    monitor_prompt: str
    monitor_interval_s: float
    instrument_monitor_interval_s: float
    instrument_state_interval_s: float
    instrument_lost_after_s: float
    monitor_history_size: int
    frame_max_age_s: float
    frame_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None


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


def _prompt(
    data: dict[str, Any],
    config_path: Path | None,
    name: str,
) -> str:
    inline = data.get(name)
    if inline is not None:
        return str(inline)
    configured = data.get(f"{name}_file")
    path = _resolve(config_path, str(configured)) if configured is not None else _PACKAGE / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


def load_config(path: Path | None) -> WorkerConfig:
    """Load YAML and resolve every path relative to that file."""

    data = _read_config(path)
    interval = float(data.get("monitor_interval_s", 5.0))
    instrument_interval = float(data.get("instrument_monitor_interval_s", interval))
    instrument_state_interval = float(data.get("instrument_state_interval_s", 10.0))
    instrument_lost_after = float(data.get("instrument_lost_after_s", 30.0))
    history_size = int(data.get("monitor_history_size", 20))
    if interval <= 0:
        raise ValueError("monitor_interval_s must be positive")
    if instrument_interval <= 0:
        raise ValueError("instrument_monitor_interval_s must be positive")
    if instrument_state_interval <= 0:
        raise ValueError("instrument_state_interval_s must be positive")
    if instrument_lost_after <= 0:
        raise ValueError("instrument_lost_after_s must be positive")
    if history_size <= 0:
        raise ValueError("monitor_history_size must be positive")
    idle_timeout = float(data.get("idle_timeout_secs", 0.0))
    return WorkerConfig(
        models_config=_resolve(path, str(data.get("models_config", "models.local.json"))),
        voice_gate_yaml=_resolve(path, str(data.get("voice_gate_yaml", "voice_gate.yaml"))),
        device_map=load_device_map(_resolve(path, str(data.get("device_map_yaml", "device_map.yaml")))),
        artifacts_dir=_resolve(path, str(data.get("artifacts_dir", "../artifacts"))),
        foreground_prompt=_prompt(data, path, "foreground_prompt"),
        monitor_prompt=_prompt(data, path, "monitor_prompt"),
        monitor_interval_s=interval,
        instrument_monitor_interval_s=instrument_interval,
        instrument_state_interval_s=instrument_state_interval,
        instrument_lost_after_s=instrument_lost_after,
        monitor_history_size=history_size,
        frame_max_age_s=float(data.get("frame_max_age_s", 5.0)),
        frame_timeout_s=float(data.get("frame_timeout_s", 5.0)),
        silence_duration=float(data.get("silence_duration", 0.4)),
        min_speech=float(data.get("min_speech", 0.1)),
        silero_threshold=float(data.get("silero_threshold", 0.5)),
        idle_timeout_secs=idle_timeout if idle_timeout > 0 else None,
    )


__all__ = ["WorkerConfig", "load_config"]
