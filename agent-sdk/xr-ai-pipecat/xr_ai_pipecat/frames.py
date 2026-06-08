# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame types unique to the xr-ai-pipecat unified voice pipeline.

Everything pipecat already ships (``InputAudioRawFrame``,
``OutputAudioRawFrame``, ``TranscriptionFrame``, ``UserStartedSpeakingFrame``,
``UserStoppedSpeakingFrame``, ``InterruptionFrame``, ``TextFrame``) is reused
directly — only the participant lifecycle and the voicegate-emitted query
frame live here.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class ParticipantJoinedFrame(DataFrame):
    """A participant joined the conversation.

    Consumed by ``VoiceGateProcessor`` (greeting hook) and
    ``BrainProcessor`` (per-pid setup). The transport adapter is
    responsible for emitting one of these per participant.
    """

    participant_id: str


@dataclass
class ParticipantLeftFrame(DataFrame):
    """A participant left the conversation.

    Consumed by ``BrainProcessor`` (per-pid teardown).
    """

    participant_id: str


@dataclass
class GatedQueryFrame(DataFrame):
    """An STT transcript that has passed the voice gate.

    Emitted by ``VoiceGateProcessor`` for the brain to consume.
    ``fresh_match`` distinguishes a fresh magic-phrase match (case 2 in
    the gate's event ladder) from a follow-up-window continuation (case
    3) so downstream can suppress one-shot side effects on follow-ups.
    """

    participant_id: str
    text: str
    fresh_match: bool
    pts_us: int


@dataclass
class BrainResponseEndFrame(DataFrame):
    """A single brain turn finished emitting ``TextFrame``s.

    Emitted by :class:`BrainProcessor` in :meth:`_run_query`'s finally
    block. Carries ``text`` — the full assembled response — so the
    downstream :class:`StreamingTtsProcessor` can echo the per-turn
    response on the data channel exactly once, matching the
    pre-migration "send full response at end" behavior. ``pid`` is the
    participant whose turn ended; the data echo addresses the same pid.

    A turn that was cancelled (new query, interruption) still produces
    one of these on the way out so the consumer never sees an open turn
    without a corresponding end marker.
    """

    pid: str
    text: str
    pts_us: int
