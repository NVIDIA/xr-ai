# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private YAML settings and phrase matching for speaker enrollment."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from urllib.parse import urlsplit

from ._conversation import _ConversationConfig


@dataclass(frozen=True)
class _SpeakerConfig(_ConversationConfig):
    base_url: str = "http://127.0.0.1:8102"
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
        if not isinstance(cfg.base_url, str):
            raise ValueError("speaker.base_url must be an HTTP origin")
        parsed = urlsplit(cfg.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("speaker.base_url must be an HTTP origin")
        for name in ("activity_threshold", "silence_duration", "max_utterance_s"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"speaker.{name} must be a positive finite number")
        if cfg.activity_threshold >= 1:
            raise ValueError("speaker.activity_threshold must be less than one")
        if cfg.max_utterance_s <= cfg.silence_duration:
            raise ValueError("speaker.max_utterance_s must exceed silence_duration")
        return cfg if enabled else None
