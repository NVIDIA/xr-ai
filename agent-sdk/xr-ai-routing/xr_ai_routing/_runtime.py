# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind typed runtime topics and translate turn I/O within ``xr_ai_routing``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import nemo_relay

try:
    from xr_ai_interaction import Interrupted, ParticipantJoined, ParticipantLeft, ResponseOutput, UserQuery
    from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe
except ModuleNotFoundError as error:
    if error.name not in {"xr_ai_interaction", "xr_ai_runtime"}:
        raise
    raise ImportError("runtime routing requires the xr-ai-routing[runtime] extra") from error

from ._contracts import AgentTurn

if TYPE_CHECKING:
    from ._router import AgentRouter


@dataclass(slots=True)
class _RuntimeTurn:
    context: RuntimeContext


class _RuntimeRouting:
    def __init__(
        self,
        owner: AgentRouter,
        query_topic: Topic[UserQuery],
        output_topic: Topic[ResponseOutput],
        *,
        participant_joined_topic: Topic[ParticipantJoined] | None,
        participant_left_topic: Topic[ParticipantLeft] | None,
        interrupted_topic: Topic[Interrupted] | None,
    ) -> None:
        subscriptions = (
            (query_topic, UserQuery, _query_receiver),
            (participant_joined_topic, ParticipantJoined, _joined_receiver),
            (participant_left_topic, ParticipantLeft, _left_receiver),
            (interrupted_topic, Interrupted, _interrupted_receiver),
        )
        names = [topic.name for topic, _, _ in subscriptions if topic is not None]
        if len(set(names)) != len(names) or output_topic.name in names:
            raise ValueError("router input, lifecycle, and output topic names must be distinct")
        if output_topic.message_type is not ResponseOutput:
            raise TypeError("router output_topic must carry ResponseOutput")
        for topic, expected, _receiver in subscriptions:
            if topic is not None and topic.message_type is not expected:
                raise TypeError(f"router topic {topic.name!r} must carry {expected.__name__}")
        self._owner = owner
        self._output_topic = output_topic
        self._subscriptions = subscriptions

    def subscriber(self) -> Agent:
        """Bind the configured topics to one runtime agent that delivers to this router."""

        namespace: dict[str, Any] = {"__module__": __name__}
        for index, (topic, _expected, receiver) in enumerate(self._subscriptions):
            if topic is not None:
                namespace[f"receive_{index}"] = subscribe(topic)(receiver())
        subscriber_type = type("RouterRuntimeAgent", (_RouterRuntimeAgent,), namespace)
        return subscriber_type(self._owner)

    async def _query(self, request: UserQuery, ctx: RuntimeContext) -> None:
        turn = AgentTurn(_participant(ctx), ctx.metadata.message_id, request.text, timestamp_us=request.timestamp_us)
        state = _RuntimeTurn(ctx)
        await self._owner._submit(turn, emit=partial(self._output, state), runtime_turn=state)

    async def _joined(self, _event: ParticipantJoined, ctx: RuntimeContext) -> None:
        await self._owner._participant_joined(_participant(ctx))

    async def _left(self, _event: ParticipantLeft, ctx: RuntimeContext) -> None:
        await self._owner.participant_left(_participant(ctx))

    async def _interrupted(self, _event: Interrupted, ctx: RuntimeContext) -> None:
        await self._owner.interrupt(ctx.metadata.participant_id)

    @contextmanager
    def scope(self, turn: AgentTurn, state: _RuntimeTurn) -> Iterator[None]:
        ctx = state.context
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(
                "routing.turn",
                nemo_relay.ScopeType.Agent,
                input={"text": turn.query, "timestamp_us": turn.timestamp_us},
                metadata={
                    "participant_id": turn.participant_id,
                    "message_id": ctx.metadata.message_id,
                    "correlation_id": ctx.metadata.correlation_id,
                    "agent": ctx.agent_name,
                },
            ):
                yield

    async def _output(self, state: _RuntimeTurn, turn: AgentTurn, source: str, text: str) -> None:
        with nemo_relay.scope.scope(
            "routing.control-output",
            nemo_relay.ScopeType.Agent,
            metadata={"destination": source},
        ):
            await state.context.publish(
                self._output_topic,
                ResponseOutput(kind="reply", text=text, request_id=turn.turn_id, timestamp_us=turn.timestamp_us),
            )


class _RouterRuntimeAgent(Agent):
    def __init__(self, router: AgentRouter) -> None:
        super().__init__()
        self._router = router


# Each factory returns a fresh handler so ``subscribe`` never marks a shared function.
def _query_receiver() -> Any:
    async def receive(self: _RouterRuntimeAgent, message: UserQuery, ctx: RuntimeContext) -> None:
        self._router._check_lifecycle_caller()
        await self._router._runtime._query(message, ctx)

    return receive


def _joined_receiver() -> Any:
    async def receive(self: _RouterRuntimeAgent, message: ParticipantJoined, ctx: RuntimeContext) -> None:
        self._router._check_lifecycle_caller()
        await self._router._runtime._joined(message, ctx)

    return receive


def _left_receiver() -> Any:
    async def receive(self: _RouterRuntimeAgent, message: ParticipantLeft, ctx: RuntimeContext) -> None:
        self._router._check_lifecycle_caller()
        await self._router._runtime._left(message, ctx)

    return receive


def _interrupted_receiver() -> Any:
    async def receive(self: _RouterRuntimeAgent, message: Interrupted, ctx: RuntimeContext) -> None:
        self._router._check_lifecycle_caller()
        await self._router._runtime._interrupted(message, ctx)

    return receive


def _participant(ctx: RuntimeContext) -> str:
    participant_id = ctx.metadata.participant_id
    if participant_id is None:
        raise ValueError("router runtime events require a participant")
    return participant_id
