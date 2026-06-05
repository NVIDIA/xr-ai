# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass, YAML loader, and consumer-facing Protocols
for the voice gate."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Protocol

import yaml


@dataclass(frozen=True)
class VoiceGateConfig:
    """Voice-gate behaviour knobs.

    ``magic_phrases``    — strict-prefix opt-in words; empty tuple disables
                           the gate so every STT transcript is dispatched.
    ``followup_grace_s`` — seconds after a phrase match during which the
                           next utterance from the same participant
                           bypasses the gate.
    ``listening_chime``  — when true AND ``magic_phrases`` is non-empty,
                           a short two-tone chime plays on the consumer's
                           audio sink whenever the worker invokes
                           ``VoiceGate.play_chime``. Defaults to ``True``
                           because the chime is an audible "I heard you"
                           cue that most consumers want by default;
                           opt out explicitly with ``listening_chime: false``.
    """
    magic_phrases:    tuple[str, ...] = ()
    followup_grace_s: float           = 5.0
    listening_chime:  bool            = True


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
        listening_chime  = bool(raw.get("listening_chime", True)),
    )


class AudioSink(Protocol):
    """Consumer-supplied return-audio writer."""

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None: ...


class TTSLike(Protocol):
    """Duck-typed text-to-speech client used for ``say_stop_ack``."""

    async def synthesize(self, text: str) -> bytes: ...
