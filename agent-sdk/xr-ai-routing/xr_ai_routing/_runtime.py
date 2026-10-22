# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind typed runtime topics and translate turn I/O within ``xr_ai_routing``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any

import nemo_relay
from xr_ai_runtime import RuntimeContext, Topic, subscribe

try:
    from xr_ai_voice import UserQuery, VoiceInterrupted, VoiceOutput, VoiceParticipantJoined, VoiceParticipantLeft
except ModuleNotFoundError as error:
    if error.name != "xr_ai_voice":
        raise
    raise ImportError("runtime routing requires the xr-ai-routing[voice] extra") from error

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
        output_topic: Topic[VoiceOutput],
        *,
        participant_joined_topic: Topic[VoiceParticipantJoined] | None,
        participant_left_topic: Topic[VoiceParticipantLeft] | None,
        interrupted_topic: Topic[VoiceInterrupted] | None,
    ) -> None:
        subscriptions = (
            (query_topic, UserQuery, self._query),
            (participant_joined_topic, VoiceParticipantJoined, self._joined),
            (participant_left_topic, VoiceParticipantLeft, self._left),
            (interrupted_topic, VoiceInterrupted, self._interrupted),
        )
        names = [topic.name for topic, _, _ in subscriptions if topic is not None]
        if len(set(names)) != len(names) or output_topic.name in names:
            raise ValueError("router input, lifecycle, and output topic names must be distinct")
        if output_topic.message_type is not VoiceOutput:
            raise TypeError("router output_topic must carry VoiceOutput")
        self._owner = owner
        self._output_topic = output_topic
        for index, (topic, expected, handler) in enumerate(subscriptions):
            if topic is not None:
                if topic.message_type is not expected:
                    raise TypeError(f"router topic {topic.name!r} must carry {expected.__name__}")
                self._bind_subscription(index, topic, handler)

    def _bind_subscription(self, index: int, topic: Topic[Any], handler: Callable[..., Awaitable[None]]) -> None:
        async def receive(_owner: AgentRouter, message: Any, context: RuntimeContext) -> None:
            _owner._check_lifecycle_caller()
            await handler(message, context)

        # Runtime discovery requires the exact configured model on each bound method.
        receive.__annotations__ = {"message": topic.message_type, "context": RuntimeContext, "return": None}
        setattr(self._owner, f"_runtime_subscription_{index}", MethodType(subscribe(topic)(receive), self._owner))

    async def _query(self, request: UserQuery, ctx: RuntimeContext) -> None:
        turn = AgentTurn(_participant(ctx), ctx.metadata.message_id, request.text, timestamp_us=request.timestamp_us)
        state = _RuntimeTurn(ctx)
        await self._owner._submit(turn, emit=partial(self._output, state), runtime_turn=state)

    async def _joined(self, _event: VoiceParticipantJoined, ctx: RuntimeContext) -> None:
        await self._owner._participant_joined(_participant(ctx))

    async def _left(self, _event: VoiceParticipantLeft, ctx: RuntimeContext) -> None:
        await self._owner.participant_left(_participant(ctx))

    async def _interrupted(self, _event: VoiceInterrupted, ctx: RuntimeContext) -> None:
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
            await state.context.publish(self._output_topic, VoiceOutput(text=text, timestamp_us=turn.timestamp_us))


def _participant(ctx: RuntimeContext) -> str:
    participant_id = ctx.metadata.participant_id
    if participant_id is None:
        raise ValueError("router runtime events require a participant")
    return participant_id
