# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the application settings that app.py uses to construct the worker.

Read shared frame and speech settings from worker.yaml, locate the model and
voice-gate configuration, and load capture-specific settings through
capture/config.py. Apply the launcher's optional output-directory override.

Keep each YAML's paths anchored to that file so launch-time overrides do not
relocate model, voice, or capture configuration.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .capture.config import CaptureConfig, load_capture_config


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Shared worker settings plus the capture capability's own configuration."""

    models_config: Path
    voice_gate_yaml: Path
    capture: CaptureConfig
    frame_max_age_s: float = 2.0
    frame_timeout_s: float = 3.0
    silence_duration: float = 0.6
    min_speech: float = 0.15
    silero_threshold: float = 0.4


def load_config(path: Path | None) -> WorkerConfig:
    """Read worker YAML or a launcher request into settings for app.run_app()."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "yaml/worker.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    output_root = None
    if "worker_config" in data:
        # Launch requests carry only an output override; settings still resolve
        # beside the original YAML, not beside the temporary request file.
        output_root = Path(data["output_root"])
        path = Path(data["worker_config"])
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent
    settings = {}
    for name in (
        "frame_max_age_s",
        "frame_timeout_s",
        "silence_duration",
        "min_speech",
        "silero_threshold",
    ):
        value = float(data[name]) if name in data else None
        if value is not None:
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
            settings[name] = value
    return WorkerConfig(
        models_config=base / "models.json",
        voice_gate_yaml=base / "voice_gate.yaml",
        capture=load_capture_config(
            base / data.get("capture_config", "capture-agent-config.yaml"), output_root=output_root,
        ),
        **settings,
    )
