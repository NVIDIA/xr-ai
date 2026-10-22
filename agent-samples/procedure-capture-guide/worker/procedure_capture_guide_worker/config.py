# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the application settings that app.py uses to construct the worker.

Read shared frame and speech settings from worker.yaml, locate the model and
voice-gate configuration, and load only the selected capability's settings through
capture/config.py or guide/config.py. Apply the launcher's mode and overrides.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .capture.config import CaptureConfig, load_capture_config
from .guide.config import GuideConfig, load_guide_config


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Shared worker settings and the configuration for the selected capability."""

    models_config: Path
    voice_gate_yaml: Path
    capture: CaptureConfig | None = None
    guide: GuideConfig | None = None
    mode: Literal["capture", "guide"] = "capture"
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
    overrides = {}
    if "worker_config" in data:
        # Launch requests carry mode and overrides; settings still resolve
        # beside the original YAML, not beside the temporary request file.
        overrides = data
        output_root = Path(data["output_root"]) if data.get("output_root") is not None else None
        path = Path(data["worker_config"])
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent
    mode = overrides.get("mode", data.get("mode", "capture"))
    if mode not in {"capture", "guide"}:
        raise ValueError("mode must be capture or guide")
    if mode == "guide" and output_root is not None:
        raise ValueError("--output-root is only available in capture mode")
    if mode == "capture" and overrides.get("guide_name") is not None:
        raise ValueError("GUIDE_NAME is only available in guide mode")
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
        mode=mode,
        capture=load_capture_config(
            base / data.get("capture_config", "capture-agent-config.yaml"), output_root=output_root,
        ) if mode == "capture" else None,
        guide=load_guide_config(
            base / data.get("guide_config", "guide-agent-config.yaml"), guide_name=overrides.get("guide_name"),
        ) if mode == "guide" else None,
        **settings,
    )
