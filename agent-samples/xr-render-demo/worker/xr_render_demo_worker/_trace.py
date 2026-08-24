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

current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")
current_participant_id: ContextVar[str] = ContextVar("current_participant_id", default="")
current_reference_time_us: ContextVar[int] = ContextVar("current_reference_time_us", default=0)


class MutationEvidence:
    """Count of scene writes actually applied (or found already satisfied)
    this turn; the supervisor's success gate reads it instead of trusting
    the model's wording."""

    def __init__(self) -> None:
        self.applied = 0
        self.satisfied = 0


current_mutation_evidence: ContextVar[MutationEvidence | None] = ContextVar(
    "current_mutation_evidence", default=None
)
