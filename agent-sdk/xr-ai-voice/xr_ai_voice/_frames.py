# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame types unique to the xr-ai-voice unified voice pipeline.

Everything pipecat already ships (``InputAudioRawFrame``,
``OutputAudioRawFrame``, ``TranscriptionFrame``, ``UserStartedSpeakingFrame``,
``UserStoppedSpeakingFrame``, ``InterruptionFrame``, ``TextFrame``) is reused
directly — only participant lifecycle, voice-gate queries, and synthesized-text
response boundaries live here.
"""
from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class ParticipantJoinedFrame(DataFrame):
    """A participant joined the conversation.

    Consumed by ``VoiceGateProcessor`` (greeting hook) and
    the private assistant processor (per-pid setup). The transport adapter is
    responsible for emitting one of these per participant.
    """

    participant_id: str


@dataclass
class ParticipantLeftFrame(DataFrame):
    """A participant left the conversation.

    Consumed by the private assistant processor (per-pid teardown).
    """

    participant_id: str


@dataclass
class GatedQueryFrame(DataFrame):
    """An STT transcript that has passed the voice gate.

    Emitted by ``VoiceGateProcessor`` for the assistant to consume.
    ``fresh_match`` distinguishes a fresh magic-phrase match (case 2 in
    the gate's event ladder) from a follow-up-window continuation (case
    3) so downstream can suppress one-shot side effects on follow-ups.
    """

    participant_id: str
    text: str
    fresh_match: bool
    pts_us: int


@dataclass
class TextResponseEndFrame(DataFrame):
    """One participant-scoped sequence of synthesized text has ended.

    Every producer that emits ``TextFrame`` objects for one utterance follows
    them with this frame. ``StreamingTtsProcessor`` uses it to flush trailing
    text and place the audio boundary behind all synthesis already queued for
    the participant.
    """

    pid: str


@dataclass
class AssistantResponseEndFrame(TextResponseEndFrame):
    """A single assistant turn finished emitting ``TextFrame``s.

    Emitted by the private assistant processor when a query completes
    block. Carries ``text`` — the full assembled response — so the
    downstream :class:`StreamingTtsProcessor` can echo the per-turn
    response on the data channel exactly once. ``pid`` is the
    participant whose turn ended; the data echo addresses the same pid.

    Only a turn that ran to completion emits one. A cancelled turn (new
    query, interruption) deliberately does not: its response is partial,
    and echoing it as the turn's final data would publish text the agent
    was interrupted out of saying.
    """

    text: str
    pts_us: int
