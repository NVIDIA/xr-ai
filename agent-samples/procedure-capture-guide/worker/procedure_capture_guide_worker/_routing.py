# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route procedure requests to capture or a pinned approved guide.

The worker composition uses this frontend only when no mode is selected.
Destinations adapt the existing agents; their backends still own capture writes,
guide progress, and verification. One voice input path owns routed response tasks.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from xr_ai_hub import ParticipantEvent
from xr_ai_models import DecisionService
from xr_ai_routing import Admission, AgentRouter, AgentTurn, DestinationResult, RoutingAid
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceAggregationAgent, VoiceInterrupted, VoiceOutput

from .capture import CaptureAgent
from .guide import GuideAgent, SelectedGuide

ROUTED_QUERY_TOPIC = Topic("procedure.router.query", UserQuery)
ROUTED_INTERRUPT_TOPIC = Topic("procedure.router.interrupted", VoiceInterrupted)
_Emit = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True)
class _Input:
    query: UserQuery
    context: RuntimeContext


class _CaptureDestination:
    def __init__(self, agent: CaptureAgent, inputs: dict[str, _Input], guide_active: Callable[[str], bool],
                 prepare: Callable[[str, str], Awaitable[None]]) -> None:
        self.agent, self._inputs, self._guide_active = agent, inputs, guide_active
        self._prepare = prepare

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid(
            name="capture",
            description=(
                "Start capturing a NEW demonstrated procedure when the user explicitly asks to capture or record "
                "their task. Selecting this route starts capture immediately. Not requests to follow an existing "
                "guide, explanations, hypothetical or negated requests. During capture all speech is narration "
                "except the exact start capture and stop capture commands; never hand narration to another guide."
            ),
            eligible=not self._guide_active(participant_id),
            eligibility="Stop the active guide before starting capture.",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        return Admission("decline" if self._guide_active(turn.participant_id) else "accept")

    async def answer(self, turn: AgentTurn, *, emit: _Emit) -> DestinationResult:
        incoming = self._inputs[turn.turn_id]
        was_active = self.agent.has_capture(turn.participant_id)
        if not was_active:
            await self._prepare(turn.participant_id, "capture")
        # Preserve STT time at the capture boundary; the router's ID is not a timestamp.
        query = incoming.query if was_active or self.agent.command(turn.query) == "stop" else UserQuery(
            text="start capture", timestamp_us=incoming.query.timestamp_us,
        )
        await self.agent.on_input(query, incoming.context)
        if self.agent.command(query.text) == "stop":
            # The backend owns finalization. Release routing focus atomically
            # with accepting stop, not after a cancellable filesystem wait.
            return DestinationResult("release", "Capture stopped." if was_active else "")
        await self.agent.drain(turn.participant_id)
        active = self.agent.has_capture(turn.participant_id)
        # CaptureAgent delivers ordered acknowledgements after backend acceptance.
        text = "Capture started." if active and not was_active else ""
        return DestinationResult("focus" if active else "release", text)

    async def release(self, participant_id: str) -> None:
        # Participant and worker cleanup drain the independently owned capture queue.
        return


class _GuideDestination:
    def __init__(
        self, selected: SelectedGuide, agent: GuideAgent, owner: Callable[[str], str | None],
        prepare: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self.selected, self.agent, self._owner = selected, agent, owner
        self._prepare = prepare
        self.name = f"guide:{selected.workflow.id}"

    def routing_aid(self, participant_id: str) -> RoutingAid:
        workflow = self.selected.workflow
        owner = self._owner(participant_id)
        return RoutingAid(
            name=self.name,
            description=(
                f"Start the approved guide {workflow.name!r} (ID {workflow.id!r}) for an explicit request "
                f"to follow this procedure. Task: {workflow.foreground_prompt}. "
                "Names need not match literally, but the requested task must match. "
                "Do not select for an unspecified procedure when multiple guides fit; use help for clarification. "
                "Do not start for explanations, hypothetical or negated requests. "
                + ("This guide is ACTIVE: handle its questions, next/skip, restart and stop guide controls."
                   if self.agent.is_active(participant_id) else "This guide is inactive.")
            ),
            eligible=workflow.runnable and owner in (None, self.name),
            eligibility=f"Current activity: {owner or 'idle'}. Stop it before changing procedure.",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        return Admission("accept" if self.routing_aid(turn.participant_id).eligible else "decline")

    async def answer(self, turn: AgentTurn, *, emit: _Emit) -> DestinationResult:
        if not self.routing_aid(turn.participant_id).eligible:
            raise RuntimeError("guide ownership changed after admission")
        if self.agent.is_active(turn.participant_id):
            text = await self.agent.respond(turn.participant_id, turn.query)
        else:
            await self._prepare(turn.participant_id, self.name)
            message = await self.agent.start_selected(turn.participant_id)
            text = f"Starting {self.selected.workflow.name}. {message}" if message else ""
        if text:
            await emit(self.name, text)
        # Re-evaluate eligibility on each turn: completion can happen in the
        # observer between turns, without leaving stale router focus behind.
        return DestinationResult("answer", text)

    async def release(self, participant_id: str) -> None:
        await self.agent.release(participant_id)


class _HelpDestination:
    def __init__(self, guides: tuple[SelectedGuide, ...], owner: Callable[[str], str | None]) -> None:
        self._names = ", ".join(guide.workflow.name for guide in guides)
        self._owner = owner

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid(
            name="help",
            description=(
                "Help, list available guides, missing or ambiguous guide selection, unavailable procedures, "
                "and any request that does not explicitly start capture or a matching approved guide. "
                "Choose this for explanations, hypothetical or negated starts. "
                "If several guides match and the user has not distinguished them, ask which one. "
                "An answer to a previous guide-choice question can select that specific guide directly."
            ),
            eligible=self._owner(participant_id) is None,
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, emit: _Emit) -> DestinationResult:
        text = "You can start capture for a new demonstration. "
        text += (f"Available approved guides: {self._names}. Which guide would you like?"
                 if self._names else "There are no approved guides available.")
        await emit("help", text)
        return DestinationResult("answer", text)

    async def release(self, participant_id: str) -> None:
        return


class ProcedureRouterAgent(Agent):
    """Connect voice turns to one capture route and one route per approved guide.

    Participant departure invalidates accepted capture starts immediately, then
    drains routing and backend work. Response cancellation never cancels queued
    capture writes or substitutes a new connection's transcript timestamps.
    """

    def __init__(
        self, *, decisions: DecisionService, capture: CaptureAgent,
        guides: tuple[tuple[SelectedGuide, GuideAgent], ...], aggregation: VoiceAggregationAgent,
    ) -> None:
        super().__init__()
        self._capture, self._aggregation = capture, aggregation
        self._inputs: dict[str, _Input] = {}
        self._connections: dict[str, ParticipantEvent] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._turns: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._guides = tuple(
            _GuideDestination(selected, agent, self._owner, self._prepare) for selected, agent in guides
        )
        self._router = AgentRouter({
            "capture": _CaptureDestination(capture, self._inputs, lambda pid: self._guide_owner(pid) is not None,
                                           self._prepare),
            **{guide.name: guide for guide in self._guides},
            "help": _HelpDestination(tuple(selected for selected, _ in guides), self._owner),
        }, decisions, default_destination="help")

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        for guide in self._guides:
            guide.agent.bind_runtime(runtime)

    def _guide_owner(self, participant_id: str) -> str | None:
        return next((guide.name for guide in self._guides if guide.agent.is_active(participant_id)), None)

    def _owner(self, participant_id: str) -> str | None:
        return "capture" if self._capture.has_capture(participant_id) else self._guide_owner(participant_id)

    async def _prepare(self, participant_id: str, destination: str) -> None:
        # Automatic completion may still be publishing its final notice. Drain
        # old observers before clearing output and starting another capability.
        for guide in self._guides:
            if guide.name != destination:
                await guide.release(participant_id)
        await self._aggregation.release(participant_id)

    async def _cancel_turn(self, participant_id: str) -> None:
        task = self._turns.pop(participant_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def receive(self, event: ParticipantEvent) -> Awaitable[None]:
        pid = event.participant_id
        previous = self._connections.get(pid)
        duplicate = previous is not None and previous.participant_session_id == event.participant_session_id
        if not self._closed:
            if event.joined and not duplicate:
                self._connections[pid] = event
            elif not event.joined and duplicate:
                self._connections.pop(pid, None)
        departing = previous is not None and not (event.joined and duplicate) and (event.joined or duplicate)
        if departing:
            for guide in self._guides:
                guide.agent.disconnect(pid)
        capture_event = self._capture.receive(event)

        async def apply() -> None:
            async with self._locks.setdefault(pid, asyncio.Lock()):
                try:
                    if departing:
                        await self._cancel_turn(pid)
                        await self._router.participant_left(pid)
                        await self._aggregation.release(pid)
                finally:
                    await capture_event

        return apply()

    @subscribe(ROUTED_QUERY_TOPIC)
    async def user_query(self, query: UserQuery, ctx: RuntimeContext) -> None:
        pid = ctx.metadata.participant_id
        if pid is None:
            return
        connection = self._connections.get(pid)
        async with self._locks.setdefault(pid, asyncio.Lock()):
            if self._closed or connection is None or query.timestamp_us < connection.pts_us:
                return
            if self._capture.has_capture(pid) and not self._capture.accepting_narration(pid):
                await self._capture.drain(pid)
            await self._cancel_turn(pid)
            await self._aggregation.release(pid)
            if self._connections.get(pid) is not connection:
                return
            self._turns[pid] = asyncio.create_task(self._answer(query, ctx, connection))

    async def _answer(self, query: UserQuery, ctx: RuntimeContext, connection: ParticipantEvent) -> None:
        pid = connection.participant_id
        turn_id = uuid.uuid4().hex
        self._inputs[turn_id] = _Input(query, ctx)

        async def emit(_source: str, text: str) -> None:
            if self._connections.get(pid) is connection:
                await ctx.publish(VOICE_OUTPUT_TOPIC, VoiceOutput(text=text, timestamp_us=query.timestamp_us),
                                  participant_id=pid)

        try:
            await self._router.handle(AgentTurn(pid, turn_id, query.text), emit=emit)
        except Exception:  # noqa: BLE001
            # isolation boundary: report a failed routed response without stopping other participants.
            logger.exception("Procedure routing failed pid={!r}", pid)
            await emit("help", "I couldn't complete that request. Please try again.")
        finally:
            self._inputs.pop(turn_id, None)

    @subscribe(ROUTED_INTERRUPT_TOPIC)
    async def interrupted(self, _event: VoiceInterrupted, ctx: RuntimeContext) -> None:
        pid = ctx.metadata.participant_id
        for participant in tuple(self._turns) if pid is None else (pid,):
            async with self._locks.setdefault(participant, asyncio.Lock()):
                await self._cancel_turn(participant)
                await self._aggregation.release(participant)

    async def stop(self) -> None:
        self._closed = True
        for pid in self._connections:
            for guide in self._guides:
                guide.agent.disconnect(pid)
        self._connections.clear()
        # Capture.close invalidates queued starts before any potentially slow drain.
        capture_close = asyncio.create_task(self._capture.close())
        try:
            for pid in tuple(self._turns):
                await self._cancel_turn(pid)
            await self._router.stop()
        finally:
            try:
                await asyncio.gather(*(guide.agent.stop() for guide in self._guides))
            finally:
                await asyncio.shield(capture_close)
