# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass and consumer-facing Protocols for the voice gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
                           ``VoiceGate.play_chime``.
    """
    magic_phrases:    tuple[str, ...] = ()
    followup_grace_s: float           = 5.0
    listening_chime:  bool            = False


class AudioSink(Protocol):
    """Consumer-supplied return-audio writer."""

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None: ...


class TTSLike(Protocol):
    """Duck-typed text-to-speech client used for ``say_stop_ack``."""

    async def synthesize(self, text: str) -> bytes: ...
