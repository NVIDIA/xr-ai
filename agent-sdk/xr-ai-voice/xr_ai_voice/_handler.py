# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-facing contracts for participant-aware voice turns."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class VoiceQuery:
    """One participant query produced by speech or typed input."""

    participant_id: str
    text: str
    fresh_match: bool
    #: Unix-epoch microseconds anchoring the query — the utterance PTS carried
    #: from the hub for spoken input, or ``time.time_ns() // 1_000`` at submit
    #: time for typed input. Use it to anchor time-relative tool calls (e.g. a
    #: "what did I just show you" recorded-frame lookup) to when the user spoke.
    timestamp_us: int


VoiceResponse: TypeAlias = str | AsyncIterator[str]
VoiceHandler: TypeAlias = Callable[[VoiceQuery], Awaitable[VoiceResponse]]


@dataclass(frozen=True, slots=True)
class VoiceTurn:
    """One completed user or agent turn observed by a voice session."""

    participant_id: str
    role: Literal["user", "agent"]
    #: Unix-epoch microseconds for the turn — the originating query's
    #: ``timestamp_us`` (both the user and the agent turn of one exchange share
    #: it, so a transcript orders the pair deterministically).
    timestamp_us: int
    text: str


__all__ = ["VoiceHandler", "VoiceQuery", "VoiceResponse", "VoiceTurn"]
