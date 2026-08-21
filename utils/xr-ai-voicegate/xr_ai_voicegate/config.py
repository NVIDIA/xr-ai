# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass, YAML loader, and consumer-facing Protocols
for the voice gate."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Protocol

import yaml

_TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
_FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}


def _parse_config_bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_STRINGS:
            return True
        if normalized in _FALSE_BOOL_STRINGS:
            return False
    accepted = sorted(_TRUE_BOOL_STRINGS | _FALSE_BOOL_STRINGS)
    raise ValueError(
        f"{key} must be a boolean or one of {accepted} (got {value!r})"
    )


@dataclass(frozen=True)
class VoiceGateConfig:
    """Voice-gate behavior settings."""

    magic_phrases:    tuple[str, ...] = ()
    """Sentence-boundary opt-in phrases.

    A match is valid at transcript start or after ``.``, ``?``, or ``!``.
    An empty tuple dispatches every non-STOP STT transcript.
    """

    followup_grace_s: float           = 5.0
    """Seconds in which the next utterance may begin without another phrase.

    The utterance may finish after the grace period.
    """

    listening_chime:  bool            = True
    """Whether phrase matches may emit a listening chime.

    The chime is available only when ``magic_phrases`` is non-empty and plays
    when the consumer calls ``VoiceGate.play_chime``. It defaults to true;
    set ``listening_chime: false`` to disable it.
    """


def load_voice_gate_config(path: pathlib.Path) -> VoiceGateConfig:
    """Load + parse a voice_gate YAML file into a :class:`VoiceGateConfig`.

    Schema: a top-level mapping with keys ``magic_phrases`` (list[str] or
    bare str), ``listening_chime`` (bool), ``followup_grace_s`` (float).
    Missing file or empty file → returns the dataclass defaults (gate
    disabled / always-on). ``magic_phrases: null`` and trailing whitespace
    in phrases are normalized the same way the inline-block parser did.
    """
    if not path.exists():
        return VoiceGateConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    phrases_raw = raw.get("magic_phrases") or []
    if isinstance(phrases_raw, str):
        phrases_raw = [phrases_raw]
    phrases = tuple(p for p in (s.strip() for s in phrases_raw) if p)

    return VoiceGateConfig(
        magic_phrases    = phrases,
        followup_grace_s = float(raw.get("followup_grace_s", 5.0)),
        listening_chime  = _parse_config_bool(
            raw.get("listening_chime", True), "listening_chime"
        ),
    )


class AudioSink(Protocol):
    """Consumer-supplied return-audio writer."""

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        """Play WAV bytes for participant *pid*."""
        ...


class TTSLike(Protocol):
    """Duck-typed text-to-speech client used for ``say_stop_ack``."""

    async def synthesize(self, text: str) -> bytes:
        """Synthesize *text* and return WAV bytes."""
        ...
