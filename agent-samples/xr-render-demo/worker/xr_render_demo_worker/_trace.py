# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn-scoped identity and evidence shared between the supervisor and its
subagents.

Context variables, not SubagentTask fields: the task schema is filled by the
supervisor LLM, which must never be trusted to copy participant identity or
timestamps. The supervisor binds these from the runtime request; subagents
read them when selecting participant-scoped frames, memory, and scene views.
"""

from contextvars import ContextVar
from typing import Literal

current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")
current_participant_id: ContextVar[str] = ContextVar("current_participant_id", default="")
current_reference_time_us: ContextVar[int] = ContextVar("current_reference_time_us", default=0)

EvidenceField = Literal["applied", "satisfied", "observed", "observed_recorded"]


class TurnEvidence:
    """Per-turn evidence counters the supervisor's gates read instead of
    trusting the model's wording: scene writes applied (or found already
    satisfied), live camera observations made (`observed`), and recorded-frame
    observations made (`observed_recorded`); only `observed` licenses a
    present-tense perception claim."""

    def __init__(self) -> None:
        self.applied = 0
        self.satisfied = 0
        self.observed = 0
        self.observed_recorded = 0


current_turn_evidence: ContextVar[TurnEvidence | None] = ContextVar(
    "current_turn_evidence", default=None
)


def record_evidence(field: EvidenceField) -> None:
    evidence = current_turn_evidence.get()
    if evidence is not None:
        setattr(evidence, field, getattr(evidence, field) + 1)
