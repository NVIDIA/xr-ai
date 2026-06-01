# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-gate handler bundle used by the conversation loop.

Workers do not normally construct this directly — :class:`ConversationLoop`
builds it from the keyword args passed to its constructor. The dataclass
is exposed so a Pipecat-style consumer that wants the same wiring
pattern without the full loop (e.g. for a custom transport) can reuse
the binding shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable


QueryResult  = str | AsyncIterator[str]
QueryHandler = Callable[[str, str, bool], Awaitable[QueryResult]]


@dataclass
class GateBindings:
    """Handlers the conversation loop registers on its internal ``VoiceGate``.

    ``on_query`` is required; everything else is optional. The loop
    constructs this from its kwargs and uses it to set up the gate; the
    type exists so the same wiring shape can be reused outside the loop."""
    on_query:              QueryHandler
    on_stop_extra:         Callable[[str], Awaitable[None]] | None = None
    on_participant_joined: Callable[[str], Awaitable[None]] | None = None
    on_participant_left:   Callable[[str], Awaitable[None]] | None = None
