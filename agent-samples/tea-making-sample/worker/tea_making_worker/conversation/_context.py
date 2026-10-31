# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project run-owned history and live state into SDK context managers.

WorkflowStore remains authoritative. Each run has incremental historical events
and replaceable live state/evidence, while dialogue survives run closure. On
rollover, bounded run context becomes an explicitly historical dialogue handoff.
Per-participant locks serialize projection, rollover and cleanup; SDK managers
own their own compaction tasks.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xr_ai_conversation import ContextBlock, ContextManager, ConversationContextUpdate

from .._capture import _event
from ..sop._state import WorkflowEvent
from ._capture import _CompactionLanguage, _context_owner

_STATE = "sop.execution"
_EVIDENCE = "sop.evidence"
_EVENTS = "sop.events"
_HANDOFF = "sop.handoff"
_TERMINAL = {"workflow.complete", "workflow.reset", "workflow.run_closed"}


@dataclass
class _Run:
    """Keep one run's historical and replaceable live context separate."""

    identity: str
    manager: ContextManager
    live: ContextManager
    handed_off: bool = False


class _RunContexts:
    """Own run managers; execution remains authoritative in WorkflowStore.

    Locks serialize refresh, event delivery and rollover per participant. A run
    manager is closed only after its bounded historical handoff is recorded.
    Dialogue never moves into a run manager and survives normal run closure.
    """

    def __init__(self, llm, dialogue: ContextManager, *, budget: int, compaction_budget: int) -> None:
        """Configure run channels; managers are created lazily for a real run identity."""
        self._llm = llm
        self._dialogue = dialogue
        self._budget = budget
        self._compaction_budget = compaction_budget
        self._runs: dict[str, _Run] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closing = False

    def blocks(self, participant: str) -> tuple[ContextBlock, ...]:
        """Return both channel types for callers that explicitly need their union."""
        run_context = self._runs.get(participant)
        return (*run_context.manager.blocks(participant), *run_context.live.blocks(participant)) if run_context else ()

    def history_blocks(self, participant: str) -> tuple[ContextBlock, ...]:
        """Return event history, which cannot authorize current state or readings."""
        run_context = self._runs.get(participant)
        return run_context.manager.blocks(participant) if run_context else ()

    def current_blocks(self, participant: str) -> tuple[ContextBlock, ...]:
        """Return the replaceable native-state and active-step evidence snapshots."""
        run_context = self._runs.get(participant)
        return run_context.live.blocks(participant) if run_context else ()

    async def event(self, participant: str, event: WorkflowEvent, *, current_run: str | None) -> None:
        """Record a current run event and hand off terminal runs under one lock.

        Old nonterminal events cannot create a replacement run. Step entry clears live
        evidence immediately; previous observations remain only in historical events.
        """
        async with self._locks.setdefault(participant, asyncio.Lock()):
            if self._closing or event.run_id is None:
                return
            run_context = self._runs.get(participant)
            is_terminal_event = event.event in _TERMINAL
            if event.run_id != current_run and not is_terminal_event:
                return
            if is_terminal_event and run_context is not None and run_context.identity != event.run_id:
                return
            run_context = await self._ensure(participant, event.run_id)
            with _context_owner(participant):
                await run_context.manager.update(
                    participant, _EVENTS,
                    ConversationContextUpdate(text=json.dumps({
                        "run_id": event.run_id, "event": event.event,
                        "step": event.step_id, "message": event.message,
                        **({"observation": event.evidence[event.step_id]} if event.step_id in event.evidence else {}),
                    }, ensure_ascii=False)),
                    mode="incremental", timestamp_us=time.time_ns() // 1000,
                )
            # New-step evidence must start empty even before the next refresh.
            # The prior observation is still available as historical event data.
            if event.event == "step.enter":
                await self._update(run_context, participant, _EVIDENCE, {
                    "run_id": event.run_id, "observations": {},
                })
                await self._update(run_context, participant, _STATE, {
                    "run_id": event.run_id, "active": True,
                    "step": {"id": event.step_id}, "state": event.state, "operations": {},
                })
            if is_terminal_event:
                await self._update(run_context, participant, _STATE, {
                    "run_id": event.run_id, "status": event.event, "final_state": event.state,
                })
                await self._retire(participant, run_context, event.event)

    async def refresh(self, participant: str, snapshot: dict[str, Any], *, valid: Callable[[], bool]) -> bool:
        """Replace live projections only while the supplied native snapshot is current.

        Args:
            participant: Owner of the context channels.
            snapshot: Native state/evidence captured by the SOP engine.
            valid: Synchronous identity/revision guard rechecked after manager rollover.

        Returns:
            Whether the snapshot was accepted, allowing the caller to retry a stale read.
        """
        async with self._locks.setdefault(participant, asyncio.Lock()):
            if self._closing or not valid():
                return False
            snapshot_run_id = snapshot.get("run_id")
            if snapshot_run_id is None:
                run_context = self._runs.get(participant)
                if run_context is not None:
                    await self._retire(participant, run_context, "closed")
                return True
            run_context = await self._ensure(participant, snapshot_run_id)
            # _ensure can drain an outgoing compaction task. Recheck after that await.
            if not valid():
                return False
            await self._update(run_context, participant, _STATE, {
                field_name: field_value for field_name, field_value in snapshot.items() if field_name != "evidence"
            })
            await self._update(run_context, participant, _EVIDENCE, {
                "run_id": snapshot_run_id, "revision": snapshot["revision"],
                "observations": snapshot["evidence"],
            })
            return True

    async def clear(self, participant: str) -> None:
        """Close both run managers before removing participant ownership."""
        async with self._locks.setdefault(participant, asyncio.Lock()):
            run_context = self._runs.get(participant)
            if run_context is not None:
                try:
                    await run_context.manager.close()
                finally:
                    await run_context.live.close()
                self._runs.pop(participant, None)

    async def close(self) -> None:
        """Reject new projections and drain all participant managers."""
        self._closing = True
        for participant in tuple(self._locks):
            await self.clear(participant)
        self._locks.clear()

    async def _ensure(self, participant: str, identity: str) -> _Run:
        """Retire a replaced run before lazily creating its successor's channels."""
        run_context = self._runs.get(participant)
        if run_context is not None and run_context.identity != identity:
            await self._retire(participant, run_context, "replaced")
            run_context = None
        if run_context is None:
            history_manager = ContextManager(
                _CompactionLanguage(self._llm),
                incremental_context_topics=(_EVENTS,),
                dialogue_budget_bytes=1,
                topic_budget_bytes=self._budget,
                compaction_input_budget_bytes=self._compaction_budget,
            )
            live_manager = ContextManager(
                _CompactionLanguage(self._llm),
                summary_context_topics=(_STATE, _EVIDENCE),
                dialogue_budget_bytes=1,
                topic_budget_bytes=self._budget,
                compaction_input_budget_bytes=self._compaction_budget,
            )
            run_context = _Run(identity, history_manager, live_manager)
            self._runs[participant] = run_context
        return run_context

    async def _retire(self, participant: str, run: _Run, reason: str) -> None:
        """Hand off bounded historical context once, then close both SDK managers.

        The caller holds the participant lock. Identity-checked removal avoids deleting a
        replacement run; the handed-off flag prevents duplicate persistent history.
        """
        if not run.handed_off:
            handoff_text = json.dumps({
                "historical_run_id": run.identity, "closed_because": reason,
                "history": [block.text for block in (*run.manager.blocks(participant), *run.live.blocks(participant))],
            }, ensure_ascii=False)
            with _context_owner(participant):
                await self._dialogue.update(
                    participant, _HANDOFF,
                    ConversationContextUpdate(text="HISTORICAL WORKFLOW HANDOFF; not current state.\n" + handoff_text),
                    mode="incremental", timestamp_us=time.time_ns() // 1000,
                )
            run.handed_off = True
            await _event("context.handoff", run_id=run.identity, reason=reason, context=handoff_text)
        try:
            await run.manager.close()
        finally:
            await run.live.close()
        if self._runs.get(participant) is run:
            self._runs.pop(participant, None)

    @staticmethod
    async def _update(run: _Run, participant: str, source: str, value: dict[str, Any]) -> None:
        """Replace one live source topic; incremental history uses a separate manager."""
        await run.live.update(
            participant, source, ConversationContextUpdate(text=json.dumps(value, ensure_ascii=False)),
            mode="summary", timestamp_us=time.time_ns() // 1000,
        )
