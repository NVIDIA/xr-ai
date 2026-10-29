# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public provider-neutral contracts for optional semantic voice admission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class TranscriptUpdate:
    """One authorized selected-speaker ASR snapshot, before query admission.

    Revisions replace the complete text of a segment. ASR finality does not
    imply semantic query completion. Source epochs change on enrollment,
    track replacement or recovery; never join text across epochs.
    """

    participant_id: str
    """Hub participant whose selected speaker produced the text."""
    source_epoch: str
    """Opaque identity of this authorized speaker-stream lifecycle."""
    segment_id: int
    """Increasing ASR segment identity within the source epoch."""
    revision: int
    """Increasing snapshot revision within the segment."""
    text: str
    """Complete current segment text, including any meaningful hesitations."""
    timestamp_us: int
    """Unix microseconds of the capture chunk associated with segment onset.

    Final snapshots use the ASR service's segment-onset timestamp. Earlier
    snapshots use the captured input chunk that delivered its start event.
    """
    final: bool = False
    """Whether ASR finalized this segment, independently of admission."""


@dataclass(frozen=True)
class VoiceOutputState:
    """Request-correlated voice output evidence, not headset playback timing."""

    participant_id: str
    """Participant receiving the response."""
    request_id: str
    """Identity supplied when the turn policy submitted the query."""
    stage: Literal["committed", "done", "interrupted"]
    """First text committed to TTS, transport output drained, or output cancelled.

    Treat committed output as non-replayable. External tool effects are owned
    and tracked by the application, independently of this voice evidence.
    """


@dataclass(frozen=True)
class VoiceTurnAction:
    """One application policy instruction applied by the voice boundary.

    Submission creates a UserQuery with this request_id. Interruption only
    applies if that request is still the current policy-admitted request.
    Queueing, merging, confidence and tool-effect rules belong to the policy.
    """

    kind: Literal["submit", "interrupt"]
    """Submit new input or cancel the matching participant's active work/audio."""
    request_id: str
    """Nonempty request identity; unique across this policy's lifecycles."""
    text: str = ""
    """Accepted query text for submit; empty for interrupt."""
    timestamp_us: int = 0
    """Capture timestamp of submitted input, in Unix microseconds."""

    def __post_init__(self) -> None:
        if (
            self.kind not in {"submit", "interrupt"}
            or not isinstance(self.request_id, str)
            or not self.request_id.strip()
        ):
            raise ValueError("invalid turn action or missing request identity")
        if not isinstance(self.text, str):
            raise ValueError("turn action text must be a string")
        if type(self.timestamp_us) is not int or self.timestamp_us < 0:
            raise ValueError("timestamp_us must be nonnegative integer microseconds")
        if (self.kind == "submit" and not self.text.strip()) or (self.kind == "interrupt" and self.text):
            raise ValueError("submit needs text; interrupt carries no text")


class VoiceTurnPolicy(Protocol):
    """Sample-owned semantic admission, injected into VoiceAgent.

    Synchronous notifications must be fast and must invalidate in-flight
    decisions when relevant state changes. decide must not start tools or
    application side effects and must propagate cancellation. The SDK owns its scheduling and drains calls
    before session cleanup. The caller owns model clients and policy lifetime.
    """

    settle_s: float
    """Finite nonnegative stable-text scheduling delay, not a silence endpoint."""

    def update(self, update: TranscriptUpdate) -> None:
        """Observe the latest authorized ASR snapshot, replacing older revisions."""
        ...

    def output(self, state: VoiceOutputState) -> None:
        """Observe request output progress and invalidate stale decisions."""
        ...

    def reset(self, participant_id: str) -> None:
        """Discard one participant's pending input, queue and decision authority."""
        ...

    async def decide(self, participant_id: str) -> Sequence[VoiceTurnAction]:
        """Return ordered actions from current state; stale results must return empty.

        Called serially across participants, with updated snapshots delivered
        synchronously while a call awaits. No repeated inference is scheduled
        for unchanged silence. Failures retain admission closed and are logged.
        """
        ...


__all__ = ["TranscriptUpdate", "VoiceOutputState", "VoiceTurnAction", "VoiceTurnPolicy"]
