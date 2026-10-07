# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private YAML settings and phrase matching for speaker enrollment."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from ._conversation import _ConversationConfig


def _default_endpoint() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    runtime_path = Path(runtime) if runtime else None
    root = (
        runtime_path / "xr-ai"
        if runtime_path is not None and runtime_path.is_absolute()
        else Path("/tmp") / f"xr-ai-{os.getuid()}"
    )
    return f"ipc://{root / 'speaker-stt.sock'}"


@dataclass(frozen=True)
class _SpeakerConfig(_ConversationConfig):
    endpoint: str = field(default_factory=_default_endpoint)
    activity_threshold: float = 0.7
    silence_duration: float = 0.6
    max_utterance_s: float = 30.0

    @classmethod
    def _from_yaml(cls, raw: object) -> _SpeakerConfig | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("speaker must be a YAML mapping")
        unknown = raw.keys() - {f.name for f in fields(cls)} - {"enabled"}
        if unknown:
            raise ValueError(f"unknown speaker settings: {sorted(unknown)}")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("speaker.enabled must be a boolean")
        cfg = cls(**{k: v for k, v in raw.items() if k != "enabled"})
        cfg._validate()
        if not isinstance(cfg.endpoint, str) or (not cfg.endpoint.startswith("ipc:///") or cfg.endpoint == "ipc:///"):
            raise ValueError("speaker.endpoint must be an absolute local ipc:// path")
        for name in ("activity_threshold", "silence_duration", "max_utterance_s"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"speaker.{name} must be a positive finite number")
        if cfg.activity_threshold >= 1:
            raise ValueError("speaker.activity_threshold must be less than one")
        if cfg.max_utterance_s <= cfg.silence_duration:
            raise ValueError("speaker.max_utterance_s must exceed silence_duration")
        return cfg if enabled else None
