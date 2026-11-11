# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latency of every ``AgentRouter`` path, driven by synthetic destinations over the real decision model.

Destinations answer instantly (or after a controlled delay), so the measured time is router overhead:
routing LLM calls, admission, acknowledgments, and control speech, not destination work.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from xr_ai_models import DecisionService, LLMService
from xr_ai_routing import (
    Admission,
    AgentRouter,
    AgentTurn,
    DestinationResult,
    RoutingAid,
)

from .._gpu import load_tags
from .._store import RunStore
from .._timing import now_ms
from .decision import _history

Dispositions = tuple[str, ...]


class _TimedDecision:
    """Delegating DecisionService that records each ``decide`` call's wall time."""

    def __init__(self, inner: DecisionService) -> None:
        self._inner = inner
        self.calls: list[float] = []

    async def decide(self, *args: Any, **kwargs: Any) -> Any:
        start = now_ms()
        try:
            return await self._inner.decide(*args, **kwargs)
        finally:
            self.calls.append(now_ms() - start)

    async def health(self) -> bool:
        return await self._inner.health()

    async def close(self) -> None:
        await self._inner.close()


class _TimedLLM:
    """Delegating LLMService that records each ``chat`` call's wall time (acknowledgments use ``chat``)."""

    def __init__(self, inner: LLMService) -> None:
        self._inner = inner
        self.chat_calls: list[float] = []
        self.capabilities = inner.capabilities

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        start = now_ms()
        try:
            return await self._inner.chat(*args, **kwargs)
        finally:
            self.chat_calls.append(now_ms() - start)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class SynthDestination:
    """A scripted destination: per-call admission dispositions and controlled latencies."""

    name: str
    description: str
    admit_script: Dispositions = ("accept",)
    admit_ms: float = 0.0
    answer_delay_ms: float = 0.0
    result: str = "answer"
    eligible: bool = True
    preamble: str = ""
    text: str = ""
    admit_times: list[float] = field(default_factory=list)
    answer_started_at: float | None = None
    answered: int = 0
    _calls: int = 0

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid(name=self.name, description=self.description, eligible=self.eligible)

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        start = now_ms()
        await asyncio.sleep(self.admit_ms / 1000.0)
        disposition = self.admit_script[min(self._calls, len(self.admit_script) - 1)]
        self._calls += 1
        self.admit_times.append(now_ms() - start)
        return Admission(disposition, preamble=self.preamble if disposition == "accept" else "", text=self.text)  # type: ignore[arg-type]

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[bool]]) -> DestinationResult:
        await asyncio.sleep(self.answer_delay_ms / 1000.0)
        await started()
        self.answer_started_at = now_ms()
        self.answered += 1
        return DestinationResult(self.result)  # type: ignore[arg-type]

    async def release(self, participant_id: str) -> None:
        return None


@dataclass
class PathSpec:
    """One router path: how to build destinations and the turn that exercises it."""

    name: str
    build: Callable[[], dict[str, SynthDestination]]
    query: str = "Please check on topic 1 for me."
    default: str = "conversation"
    prior_focus_turn: bool = False
    ack_delay_s: float | None = None
    history_bytes: int = 0


def _dests(**overrides: dict[str, Any]) -> dict[str, SynthDestination]:
    base = {
        "conversation": SynthDestination("conversation", "General conversation and small talk."),
        "topic_1": SynthDestination("topic_1", "Handles requests about topic 1: gauges and pressure readings."),
        "topic_2": SynthDestination("topic_2", "Handles requests about topic 2: scheduling and shift handover."),
    }
    for name, kw in overrides.items():
        for key, value in kw.items():
            setattr(base[name], key, value)
    return base


def paths() -> list[PathSpec]:
    """The measured router paths (names are the keys used in the report)."""
    return [
        PathSpec("single_candidate", lambda: {"conversation": SynthDestination("conversation", "General chat.")}),
        PathSpec("multi_route_accept", _dests),
        PathSpec("multi_route_accept_ctx12k", _dests, history_bytes=12_000),
        PathSpec("focused_fast_path", _dests, prior_focus_turn=True),
        PathSpec("all_ineligible_clarify", lambda: _dests(
            conversation={"eligible": False}, topic_1={"eligible": False}, topic_2={"eligible": False})),
        PathSpec("admission_clarify", lambda: _dests(
            topic_1={"admit_script": ("clarify",), "text": "Which gauge do you mean?"})),
        PathSpec("misroute_declined_once", lambda: _dests(topic_1={"admit_script": ("decline",)})),
        PathSpec("release_with_text_then_reroute", lambda: _dests(
            topic_1={"admit_script": ("release",), "text": "That is outside what I handle."})),
        PathSpec("declines_exhausted_default", lambda: _dests(
            topic_1={"admit_script": ("decline",)}, topic_2={"admit_script": ("decline",)})),
        PathSpec("preamble_then_answer", lambda: _dests(
            topic_1={"preamble": "Sure, checking the gauge now."})),
        PathSpec("admit_slow_500ms", lambda: _dests(topic_1={"admit_ms": 500.0})),
    ]


@dataclass
class RouterPlan:
    """Sample counts and the acknowledgment sweep."""

    samples: int = 15
    ack_delay_s: float = 1.0
    ack_answer_delays_ms: tuple[float, ...] = (0.0, 600.0, 1100.0, 1500.0, 1900.0, 2300.0, 2500.0, 6000.0)
    ack_emit_ms: tuple[float, ...] = (0.0, 1200.0)
    only: frozenset[str] = frozenset()


async def _one_turn(spec: PathSpec, decision: DecisionService, ack_llm: LLMService | None,
                    ack_delay_s: float | None, answer_delay_ms: float = 0.0, emit_ms: float = 0.0) -> dict[str, Any]:
    destinations = spec.build()
    for dest in destinations.values():
        if not dest.answer_delay_ms:
            dest.answer_delay_ms = answer_delay_ms
    timed_decision = _TimedDecision(decision)
    timed_llm = _TimedLLM(ack_llm) if ack_llm is not None and ack_delay_s is not None else None
    outputs: list[tuple[float, str]] = []

    async def output(turn: AgentTurn, source: str, text: str) -> None:
        outputs.append((now_ms() - t0, text))
        await asyncio.sleep(emit_ms / 1000.0)  # how long delivering control speech blocks the router

    router = AgentRouter(
        destinations, timed_decision, output=output, default_destination=spec.default,
        acknowledgment_llm=timed_llm, acknowledgment_delay_secs=ack_delay_s if timed_llm else None,
    )
    pid = f"p-{uuid.uuid4().hex[:8]}"
    history = tuple(_exchanges(spec.history_bytes))
    try:
        if spec.prior_focus_turn:
            for dest in destinations.values():
                dest.result = "focus"
            t0 = now_ms()
            await router.handle(AgentTurn(pid, "warm", spec.query, history=history))
            for dest in destinations.values():
                dest.result = "answer"
            timed_decision.calls.clear()
            for dest in destinations.values():
                dest.admit_times.clear()
                dest.answer_started_at = None
        t0 = now_ms()
        turn_history = () if spec.prior_focus_turn else history
        result = await router.handle(AgentTurn(pid, "t1", spec.query, history=turn_history))
        total = now_ms() - t0
    finally:
        await _close(router)
    started = [d.answer_started_at - t0 for d in destinations.values() if d.answer_started_at is not None]
    return {
        "total": total,
        "answer_started": min(started) if started else None,
        "first_output": outputs[0][0] if outputs else None,
        "outputs": len(outputs),
        "decide_calls": len(timed_decision.calls),
        "decide_ms": round(sum(timed_decision.calls), 2),
        "admit_ms": round(sum(sum(d.admit_times) for d in destinations.values()), 2),
        "ack_chat_calls": len(timed_llm.chat_calls) if timed_llm else 0,
        "ack_chat_ms": round(sum(timed_llm.chat_calls), 2) if timed_llm else 0.0,
        "disposition": result.disposition,
        "answered_by": next((d.name for d in destinations.values() if d.answered), None),
    }


async def _close(router: AgentRouter) -> None:
    await router.stop()


def _exchanges(nbytes: int) -> list[Any]:
    from xr_ai_routing import AgentExchange

    return [AgentExchange(user=e["user"], assistant=e["assistant"]) for e in _history(nbytes)]


async def run(decision: DecisionService, ack_llm: LLMService, store: RunStore, plan: RouterPlan) -> None:
    """Measure every path, then the acknowledgment race across destination answer delays."""
    for spec in paths():
        if plan.only and spec.name not in plan.only:
            continue
        for _ in range(plan.samples):
            metrics = await _one_turn(spec, decision, None, None)
            store.add("router.path", {"path": spec.name, **load_tags()}, **metrics)
    if plan.only and "ack" not in plan.only:
        return
    spec = PathSpec("ack_race", _dests, query="Please do a detailed review of topic 1 for me.")
    for emit_ms in plan.ack_emit_ms:
        for delay_ms in plan.ack_answer_delays_ms:
            for _ in range(plan.samples):
                metrics = await _one_turn(spec, decision, ack_llm, plan.ack_delay_s, answer_delay_ms=delay_ms,
                                          emit_ms=emit_ms)
                store.add("router.ack", {"answer_delay_ms": delay_ms, "ack_delay_s": plan.ack_delay_s,
                                         "emit_ms": emit_ms, **load_tags()}, **metrics)
