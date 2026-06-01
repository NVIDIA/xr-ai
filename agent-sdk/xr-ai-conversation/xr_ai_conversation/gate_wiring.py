# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-gate handler bundle + one-shot wiring helper.

:class:`GateBindings` is the dataclass the :class:`ConversationLoop`
builds internally from its constructor kwargs. It groups the handlers
the loop registers on its private ``VoiceGate``.

:func:`wire_voice_gate` is the public escape hatch for Pipecat-flavored
workers that own their own ``VoiceGate`` (because they have a real
frame pipeline rather than the loop's audio-callback path) but still
want to skip the five ``gate.on_*`` registration lines. The default
``on_drop`` logs at DEBUG so the gate's own drop-reason logging stays
authoritative.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

from xr_ai_voicegate import VoiceGate


logger = logging.getLogger("xr_ai_conversation.gate_wiring")


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


async def _default_drop_logger(pid: str, text: str) -> None:
    """Default ``on_drop`` handler — DEBUG-level log only.

    The voice gate already logs the drop reason at its own preferred
    level; we just emit a paired DEBUG line so the loop's logger shows
    the same event when tracing is enabled."""
    logger.debug("voice gate dropped pid=%r text=%r", pid, text[:80])


def wire_voice_gate(
    gate: VoiceGate,
    *,
    on_query:              Callable[[str, str, bool], Awaitable[None]],
    on_stop:               Callable[[str], Awaitable[None]],
    on_phrase_only:        Callable[[str], Awaitable[None]] | None = None,
    on_drop:               Callable[[str, str], Awaitable[None]] | None = None,
    on_participant_joined: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Register all five voice-gate handlers in one call.

    Pipecat-style workers that own their own ``VoiceGate`` (because they
    have a real frame pipeline rather than the loop's audio-callback
    path) use this to skip five lines of boilerplate. The
    :class:`ConversationLoop` does its own equivalent wiring internally
    and does not call this helper.

    ``on_query`` and ``on_stop`` are required — every voice-gate
    consumer at least dispatches queries and acknowledges STOP. The
    other three are optional:

    * ``on_phrase_only`` — fired when a magic phrase opens the
      follow-up window without a payload. No default (callers that
      want a chime supply ``lambda pid: gate.play_chime(pid)``).
    * ``on_drop`` — fired when the gate refused to dispatch. Defaults
      to a DEBUG log; pass ``None`` to keep the default or a callable
      to override.
    * ``on_participant_joined`` — fired once per joined participant.
      No default (the loop owns its own greeting path; pipecat workers
      typically send a custom greeting).
    """
    gate.on_query(on_query)
    gate.on_stop(on_stop)
    if on_phrase_only is not None:
        gate.on_phrase_only(on_phrase_only)
    if on_drop is not None:
        gate.on_drop(on_drop)
    else:
        gate.on_drop(_default_drop_logger)
    if on_participant_joined is not None:
        gate.on_participant_joined(on_participant_joined)
