# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-owned context channels with a historical handoff to persistent dialogue."""

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
        self._llm, self._dialogue = llm, dialogue
        self._budget, self._compaction_budget = budget, compaction_budget
        self._runs: dict[str, _Run] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closing = False

    def blocks(self, participant: str) -> tuple[ContextBlock, ...]:
        run = self._runs.get(participant)
        return (*run.manager.blocks(participant), *run.live.blocks(participant)) if run else ()

    def history_blocks(self, participant: str) -> tuple[ContextBlock, ...]:
        run = self._runs.get(participant)
        return run.manager.blocks(participant) if run else ()

    def current_blocks(self, participant: str) -> tuple[ContextBlock, ...]:
        run = self._runs.get(participant)
        return run.live.blocks(participant) if run else ()

    async def event(self, participant: str, event: WorkflowEvent, *, current_run: str | None) -> None:
        async with self._locks.setdefault(participant, asyncio.Lock()):
            if self._closing or event.run_id is None:
                return
            run = self._runs.get(participant)
            terminal = event.event in _TERMINAL
            if event.run_id != current_run and not terminal:
                return
            if terminal and run is not None and run.identity != event.run_id:
                return
            run = await self._ensure(participant, event.run_id)
            with _context_owner(participant):
                await run.manager.update(
                    participant, _EVENTS,
                    ConversationContextUpdate(text=json.dumps({
                        "run_id": event.run_id, "event": event.event,
                        "step": event.step_id, "message": event.message,
                        **({"observation": event.evidence[event.step_id]} if event.step_id in event.evidence else {}),
                    }, ensure_ascii=False)),
                    mode="incremental", timestamp_us=time.time_ns() // 1000,
                )
            if event.event == "step.enter":
                await self._update(run, participant, _EVIDENCE, {
                    "run_id": event.run_id, "observations": {},
                })
                await self._update(run, participant, _STATE, {
                    "run_id": event.run_id, "active": True,
                    "step": {"id": event.step_id}, "state": event.state, "operations": {},
                })
            if terminal:
                await self._update(run, participant, _STATE, {
                    "run_id": event.run_id, "status": event.event, "final_state": event.state,
                })
                await self._retire(participant, run, event.event)

    async def refresh(self, participant: str, snapshot: dict[str, Any], *, valid: Callable[[], bool]) -> bool:
        async with self._locks.setdefault(participant, asyncio.Lock()):
            if self._closing or not valid():
                return False
            identity = snapshot.get("run_id")
            if identity is None:
                run = self._runs.get(participant)
                if run is not None:
                    await self._retire(participant, run, "closed")
                return True
            run = await self._ensure(participant, identity)
            # _ensure can drain an outgoing compaction task. Recheck after that await.
            if not valid():
                return False
            await self._update(run, participant, _STATE, {
                k: v for k, v in snapshot.items() if k != "evidence"
            })
            await self._update(run, participant, _EVIDENCE, {
                "run_id": identity, "revision": snapshot["revision"],
                "observations": snapshot["evidence"],
            })
            return True

    async def clear(self, participant: str) -> None:
        async with self._locks.setdefault(participant, asyncio.Lock()):
            run = self._runs.get(participant)
            if run is not None:
                try:
                    await run.manager.close()
                finally:
                    await run.live.close()
                self._runs.pop(participant, None)

    async def close(self) -> None:
        self._closing = True
        for participant in tuple(self._locks):
            await self.clear(participant)
        self._locks.clear()

    async def _ensure(self, participant: str, identity: str) -> _Run:
        run = self._runs.get(participant)
        if run is not None and run.identity != identity:
            await self._retire(participant, run, "replaced")
            run = None
        if run is None:
            history = ContextManager(
                _CompactionLanguage(self._llm), incremental_context_topics=(_EVENTS,),
                dialogue_budget_bytes=1, topic_budget_bytes=self._budget,
                compaction_input_budget_bytes=self._compaction_budget,
            )
            live = ContextManager(
                _CompactionLanguage(self._llm), summary_context_topics=(_STATE, _EVIDENCE),
                dialogue_budget_bytes=1, topic_budget_bytes=self._budget,
                compaction_input_budget_bytes=self._compaction_budget,
            )
            run = _Run(identity, history, live)
            self._runs[participant] = run
        return run

    async def _retire(self, participant: str, run: _Run, reason: str) -> None:
        if not run.handed_off:
            text = json.dumps({
                "historical_run_id": run.identity, "closed_because": reason,
                "history": [block.text for block in (*run.manager.blocks(participant), *run.live.blocks(participant))],
            }, ensure_ascii=False)
            with _context_owner(participant):
                await self._dialogue.update(
                    participant, _HANDOFF,
                    ConversationContextUpdate(text="HISTORICAL WORKFLOW HANDOFF; not current state.\n" + text),
                    mode="incremental", timestamp_us=time.time_ns() // 1000,
                )
            run.handed_off = True
            await _event("context.handoff", run_id=run.identity, reason=reason, context=text)
        try:
            await run.manager.close()
        finally:
            await run.live.close()
        if self._runs.get(participant) is run:
            self._runs.pop(participant, None)

    @staticmethod
    async def _update(run: _Run, participant: str, source: str, value: dict[str, Any]) -> None:
        await run.live.update(
            participant, source, ConversationContextUpdate(text=json.dumps(value, ensure_ascii=False)),
            mode="summary", timestamp_us=time.time_ns() // 1000,
        )
