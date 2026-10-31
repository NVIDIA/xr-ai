# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give asynchronous context compaction its own participant-scoped capture lifetime."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from ._capture import _attempt

_PARTICIPANT: ContextVar[str] = ContextVar("sop_context_participant")


@contextmanager
def _context_owner(participant_id: str):
    # ContextManager schedules background tasks synchronously during an update.
    # They inherit this participant, but open their own recording attempt.
    token = _PARTICIPANT.set(participant_id)
    try:
        yield
    finally:
        _PARTICIPANT.reset(token)


class _CompactionLanguage:
    def __init__(self, llm):
        self._llm = llm

    async def chat(self, messages, **kwargs):
        async with _attempt(_PARTICIPANT.get(), "context"):
            return await self._llm.chat(messages, **kwargs)
