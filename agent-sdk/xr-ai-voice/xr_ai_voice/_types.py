# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal contracts for participant-aware voice I/O."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class VoiceQuery:
    """One participant query produced by speech or typed input."""

    participant_id: str
    text: str
    #: Unix-epoch microseconds anchoring the query to when the user spoke or typed.
    timestamp_us: int
    #: The voice consumer closed active output before accepting this replacement.
    interrupted_output: bool = False


VoiceInputSink: TypeAlias = Callable[[VoiceQuery], Awaitable[None]]
VoiceResponse: TypeAlias = str | AsyncIterator[str]


__all__ = ["VoiceInputSink", "VoiceQuery", "VoiceResponse"]
