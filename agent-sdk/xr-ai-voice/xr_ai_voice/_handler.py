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
    timestamp_us: int


VoiceResponse: TypeAlias = str | AsyncIterator[str]
VoiceHandler: TypeAlias = Callable[[VoiceQuery], Awaitable[VoiceResponse]]


@dataclass(frozen=True, slots=True)
class VoiceTurn:
    """One completed user or agent turn observed by a voice session."""

    participant_id: str
    role: Literal["user", "agent"]
    timestamp_us: int
    text: str


__all__ = ["VoiceHandler", "VoiceQuery", "VoiceResponse", "VoiceTurn"]
