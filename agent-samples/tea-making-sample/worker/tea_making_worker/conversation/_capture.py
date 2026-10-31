# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correlate SDK context-compaction calls with their participant's recording.

Compaction may outlive the foreground attempt that triggered it. A lightweight
context variable carries participant ownership so the recording wrapper can open
an independent attempt; ContextManager still owns compaction and its lifecycle.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from .._capture import _attempt

_PARTICIPANT: ContextVar[str] = ContextVar("sop_context_participant")


@contextmanager
def _context_owner(participant_id: str):
    """Scope participant identity across SDK-created background compaction tasks."""
    # ContextManager schedules background tasks synchronously during an update.
    # They inherit this participant, but open their own recording attempt.
    token = _PARTICIPANT.set(participant_id)
    try:
        yield
    finally:
        _PARTICIPANT.reset(token)


class _CompactionLanguage:
    """Delegate language calls while recording independent context-compaction attempts."""

    def __init__(self, llm):
        """Wrap the existing language service without changing its contract."""
        self._llm = llm

    async def chat(self, messages, **kwargs):
        """Record compaction with inherited participant ownership, then delegate unchanged."""
        async with _attempt(_PARTICIPANT.get(), "context"):
            return await self._llm.chat(messages, **kwargs)
