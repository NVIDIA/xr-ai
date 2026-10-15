# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private YAML settings and phrase matching for speaker enrollment."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from urllib.parse import urlsplit

from ._conversation import _ConversationConfig


@dataclass(frozen=True)
class _SpeakerConfig(_ConversationConfig):
    base_url: str = "http://127.0.0.1:8102"
    backend: str = "auto"
    timeout_s: float = 10.0
    activity_threshold: float = 0.7
    silence_duration: float = 0.6
    max_utterance_s: float = 30.0
    mode: str = "conversation"
    magic_phrases: tuple[str, ...] = ()
    followup_grace_s: float = 5.0
    stop_commands_enabled: bool = True

    def _validate_interaction(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"always_on", "wake_only", "conversation", "hybrid"}:
            raise ValueError("speaker mode must be always_on, wake_only, conversation, or hybrid")
        if not isinstance(self.magic_phrases, (list, tuple)) or not all(
            isinstance(p, str) and p.strip() for p in self.magic_phrases
        ):
            raise ValueError("speaker magic_phrases must be a sequence of non-empty strings")
        if self.mode in {"wake_only", "hybrid"} and not self.magic_phrases:
            raise ValueError("speaker wake_only and hybrid modes require magic_phrases")
        if not isinstance(self.stop_commands_enabled, bool):
            raise ValueError("speaker stop_commands_enabled must be a boolean")
        grace = self.followup_grace_s
        if isinstance(grace, bool) or not isinstance(grace, (int, float)) or not math.isfinite(grace) or grace < 0:
            raise ValueError("speaker followup_grace_s must be finite and non-negative")

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
        cfg._validate_interaction()
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
        if cfg.backend not in ("auto", "required"):
            raise ValueError("speaker.backend must be auto or required")
        cfg = replace(cfg, magic_phrases=tuple(cfg.magic_phrases))
        for name in ("activity_threshold", "silence_duration", "max_utterance_s", "timeout_s"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"speaker.{name} must be a positive finite number")
        if cfg.activity_threshold >= 1:
            raise ValueError("speaker.activity_threshold must be less than one")
        if cfg.max_utterance_s <= cfg.silence_duration:
            raise ValueError("speaker.max_utterance_s must exceed silence_duration")
        return cfg if enabled else None
