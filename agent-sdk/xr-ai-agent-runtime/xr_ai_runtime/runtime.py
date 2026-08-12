# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed event routing between agents."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar, cast, get_type_hints

from pydantic import BaseModel

from .agent import Agent
from .events import MessageMetadata, Topic

MessageT = TypeVar("MessageT", bound=BaseModel)
AgentT = TypeVar("AgentT", bound="Agent")

BoundSubscriber: TypeAlias = Callable[[BaseModel, "RuntimeContext"], Awaitable[None]]


class RuntimeClosedError(RuntimeError):
    """Raised when work is submitted to a runtime that is not running."""


@dataclass(slots=True)
class _AgentState:
    name: str
    deliveries: set[asyncio.Task[None]] = field(default_factory=set)


class RuntimeContext:
    """Runtime operations available during a subscription delivery."""

    __slots__ = ("_runtime", "_agent_name", "_metadata")

    def __init__(
        self,
        runtime: AgentRuntime,
        agent_name: str,
        metadata: MessageMetadata,
    ) -> None:
        self._runtime = runtime
        self._agent_name = agent_name
        self._metadata = metadata

    @property
    def agent_name(self) -> str:
        """Return this agent's runtime-local name."""

        return self._agent_name

    @property
    def metadata(self) -> MessageMetadata:
        """Return metadata for the current subscription delivery."""

        return self._metadata

    async def publish(
        self,
        topic: Topic[MessageT],
        message: MessageT | dict[str, Any],
        *,
        participant_id: str | None = None,
    ) -> None:
        """Publish an event while preserving current trace context."""

        await self._runtime._publish(
            topic,
            message,
            participant_id=self._resolve_participant(participant_id),
            source=self._agent_name,
            correlation_id=self._metadata.correlation_id,
            parent_message_id=self._metadata.message_id,
        )

    def _resolve_participant(self, participant_id: str | None) -> str:
        if participant_id is not None:
            return participant_id
        return self._metadata.participant_id


class AgentRuntime:
    """Provide typed pub/sub routing between registered agents."""

    def __init__(self) -> None:
        self._agents: dict[str, _AgentState] = {}
        self._topics: dict[str, Topic[Any]] = {}
        self._subscribers: dict[str, list[tuple[_AgentState, BoundSubscriber]]] = {}
        self._running = False
        self._closed = False

    @property
    def running(self) -> bool:
        """Whether the runtime currently accepts event work."""

        return self._running and not self._closed

    def register(self, name: str, agent: AgentT) -> AgentT:
        """Register one agent before startup and return the same typed object."""

        if self._running or self._closed:
            raise RuntimeError("agents must be registered before the runtime starts")
        if not name.strip():
            raise ValueError("agent name must not be empty")
        if name in self._agents:
            raise ValueError(f"agent {name!r} is already registered")
        if not isinstance(agent, Agent):
            raise TypeError("registered agents must inherit Agent")
        try:
            agent.tools
        except AttributeError as exc:
            raise TypeError("Agent subclasses must call super().__init__()") from exc

        state = _AgentState(name=name)
        subscriptions = self._discover_subscriptions(agent)
        known_topics = dict(self._topics)
        for topic, _method in subscriptions:
            known = known_topics.setdefault(topic.name, topic)
            if known.message_type is not topic.message_type:
                raise ValueError(f"topic {topic.name!r} already uses another message type")

        self._agents[name] = state
        for topic, method in subscriptions:
            self._register_topic(topic)
            self._subscribers.setdefault(topic.name, []).append((state, method))
        return agent

    async def start(self) -> None:
        """Start accepting event publications."""

        if self._closed:
            raise RuntimeClosedError("agent runtime is closed")
        if self._running:
            return
        self._running = True

    async def publish(
        self,
        topic: Topic[MessageT],
        message: MessageT | dict[str, Any],
        *,
        participant_id: str,
        source: str = "application",
    ) -> None:
        """Validate and deliver one event to every topic subscriber."""

        await self._publish(
            topic,
            message,
            participant_id=participant_id,
            source=source,
        )

    async def stop(self) -> None:
        """Stop accepting events and cancel in-flight deliveries."""

        if self._closed:
            return
        self._running = False
        tasks = tuple(
            task for state in self._agents.values() for task in state.deliveries
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._closed = True

    async def __aenter__(self) -> AgentRuntime:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def _publish(
        self,
        topic: Topic[MessageT],
        message: MessageT | dict[str, Any],
        *,
        participant_id: str,
        source: str,
        correlation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> None:
        self._ensure_running()
        self._register_topic(topic)
        value = topic.validate(message)
        metadata = self._metadata(
            participant_id=participant_id,
            source=source,
            correlation_id=correlation_id,
            parent_message_id=parent_message_id,
        )
        deliveries: list[asyncio.Task[None]] = []
        for state, method in tuple(self._subscribers.get(topic.name, ())):
            task = asyncio.create_task(
                self._deliver(
                    state,
                    method,
                    value.model_copy(deep=True),
                    metadata,
                ),
                name=f"agent:{state.name}:subscription:{topic.name}",
            )
            state.deliveries.add(task)
            task.add_done_callback(state.deliveries.discard)
            deliveries.append(task)
        if deliveries:
            results = await asyncio.gather(*deliveries, return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise BaseExceptionGroup("errors during event publication", errors)

    async def _deliver(
        self,
        state: _AgentState,
        method: BoundSubscriber,
        message: BaseModel,
        metadata: MessageMetadata,
    ) -> None:
        await method(message, RuntimeContext(self, state.name, metadata))

    def _discover_subscriptions(
        self,
        agent: Agent,
    ) -> list[tuple[Topic[Any], BoundSubscriber]]:
        subscriptions: list[tuple[Topic[Any], BoundSubscriber]] = []
        for _name, method in inspect.getmembers(agent, predicate=inspect.ismethod):
            topics = cast(tuple[Topic[Any], ...], getattr(method, "__xr_ai_topics__", ()))
            if not topics:
                continue
            if not inspect.iscoroutinefunction(method):
                raise TypeError("topic subscribers must be async")
            request_type = self._request_type(method)
            for topic in topics:
                if topic.message_type is not request_type:
                    raise TypeError(
                        f"subscriber for {topic.name!r} must accept "
                        f"{topic.message_type.__name__}"
                    )
                subscriptions.append((topic, cast(BoundSubscriber, method)))
        return subscriptions

    @staticmethod
    def _request_type(method: Callable[..., Any]) -> type[BaseModel]:
        parameters = tuple(inspect.signature(method).parameters.values())
        if len(parameters) != 2:
            raise TypeError("subscribers must accept exactly (message, context)")
        annotation = get_type_hints(method).get(parameters[0].name)
        if not isinstance(annotation, type) or not issubclass(annotation, BaseModel):
            raise TypeError("subscriber messages must be annotated Pydantic models")
        return annotation

    def _register_topic(self, topic: Topic[Any]) -> None:
        known = self._topics.setdefault(topic.name, topic)
        if known.message_type is not topic.message_type:
            raise ValueError(f"topic {topic.name!r} already uses another message type")

    def _ensure_running(self) -> None:
        if not self.running:
            raise RuntimeClosedError("agent runtime is not running")

    @staticmethod
    def _metadata(
        *,
        participant_id: str,
        source: str,
        correlation_id: str | None,
        parent_message_id: str | None,
    ) -> MessageMetadata:
        if not participant_id.strip():
            raise ValueError("participant_id must not be empty")
        if not source.strip():
            raise ValueError("message source must not be empty")
        message_id = uuid.uuid4().hex
        return MessageMetadata(
            message_id=message_id,
            correlation_id=correlation_id or message_id,
            participant_id=participant_id,
            source=source,
            parent_message_id=parent_message_id,
            timestamp_us=time.time_ns() // 1_000,
        )


__all__ = [
    "AgentRuntime",
    "RuntimeClosedError",
    "RuntimeContext",
]
