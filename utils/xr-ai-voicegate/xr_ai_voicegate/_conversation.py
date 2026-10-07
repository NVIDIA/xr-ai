# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private conversation controls shared by ordinary and speaker-conditioned STT."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields


def _normalize(text: str) -> str:
    words = re.findall(r"\w+", text.casefold().replace("'", "").replace("’", ""))
    return re.sub(r"\blet us\b", "lets", " ".join(words))


@dataclass(frozen=True)
class _ConversationConfig:
    start_phrase: str = "Hey agent, let's start talking"
    stop_phrase: str = "Hey agent, let's stop talking"
    require_wake_phrase: bool = False
    phrase_window_s: float = 6.0

    def _validate(self) -> None:
        for name in ("start_phrase", "stop_phrase"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _normalize(value):
                raise ValueError(f"conversation.{name} must contain words")
        if _normalize(self.start_phrase) == _normalize(self.stop_phrase):
            raise ValueError("conversation start and stop phrases must differ")
        if not isinstance(self.require_wake_phrase, bool):
            raise ValueError("conversation.require_wake_phrase must be a boolean")
        value = self.phrase_window_s
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("conversation.phrase_window_s must be a positive finite number")

    @classmethod
    def _from_yaml(cls, raw: object) -> _ConversationConfig | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("conversation must be a YAML mapping")
        unknown = raw.keys() - {f.name for f in fields(cls)} - {"enabled"}
        if unknown:
            raise ValueError(f"unknown conversation settings: {sorted(unknown)}")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("conversation.enabled must be a boolean")
        cfg = cls(**{k: v for k, v in raw.items() if k != "enabled"})
        cfg._validate()
        return cfg if enabled else None

    def _matches_start(self, text: str) -> bool:
        return _normalize(text) == _normalize(self.start_phrase)

    def _matches_stop(self, text: str) -> bool:
        return _normalize(text) == _normalize(self.stop_phrase)

    def _could_be_control(self, text: str) -> bool:
        words = _normalize(text)
        return any(_normalize(p).startswith(words) for p in (self.start_phrase, self.stop_phrase))


class _ControlMatcher:
    """Join only exact control-phrase prefixes from one source in a bounded window."""

    def __init__(self, cfg: _ConversationConfig) -> None:
        self.cfg = cfg
        self._key: object = None
        self._started_s = 0.0
        self._words = ""

    def _reset(self) -> None:
        self._key = None
        self._words = ""

    def _feed(self, text: str, *, key: object, at_s: float) -> str | None:
        words = _normalize(text)
        if not words:
            self._reset()
            return None
        controls = {"start": _normalize(self.cfg.start_phrase), "stop": _normalize(self.cfg.stop_phrase)}
        if self._words and (key != self._key or not 0 <= at_s - self._started_s <= self.cfg.phrase_window_s):
            self._reset()
        combined = f"{self._words} {words}" if self._words else words
        for action, phrase in controls.items():
            if words == phrase or combined == phrase:
                self._reset()
                return action
        if any(phrase.startswith(combined + " ") for phrase in controls.values()):
            if not self._words:
                self._started_s = at_s
            self._words = combined
            self._key = key
            return "pending"
        self._reset()
        if any(phrase.startswith(words + " ") for phrase in controls.values()):
            self._words = words
            self._key = key
            self._started_s = at_s
            return "pending"
        return None
