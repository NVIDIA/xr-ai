# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent resource lifetimes, background work, and typed event routing."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar, cast, get_type_hints

from pydantic import BaseModel
from xr_ai_tools import AsyncTool, Tool

MessageT = TypeVar("MessageT", bound=BaseModel)
TaskResultT = TypeVar("TaskResultT")
AgentT = TypeVar("AgentT", bound="Agent")

AgentTool: TypeAlias = Tool[Any, Any] | AsyncTool[Any, Any]
BoundSubscriber: TypeAlias = Callable[[BaseModel, "AgentContext"], Awaitable[None]]


class RuntimeClosedError(RuntimeError):
    """Raised when work is submitted to a runtime that is not running."""


class RuntimeFailedError(RuntimeError):
    """Raised when runtime-owned background work has failed."""


@dataclass(frozen=True, slots=True)
class Topic(Generic[MessageT]):
    """A stable publish/subscribe name paired with its payload model."""

    name: str
    message_type: type[MessageT]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("topic name must not be empty")
        if not issubclass(self.message_type, BaseModel):
            raise TypeError("topic messages must be Pydantic models")

    def validate(self, message: MessageT | dict[str, Any]) -> MessageT:
        """Validate a message before delivery."""

        return self.message_type.model_validate(message)


@dataclass(frozen=True, slots=True)
class MessageMetadata:
    """Routing and trace context for one published event."""

    message_id: str
    correlation_id: str
    participant_id: str
    source: str
    parent_message_id: str | None
    timestamp_us: int


def subscribe(
    topic: Topic[MessageT],
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """Register an agent method as a typed topic subscriber."""

    def decorate(method: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        topics = (*getattr(method, "__xr_ai_topics__", ()), topic)
        setattr(method, "__xr_ai_topics__", topics)
        return method

    return decorate


class Agent:
    """A runtime-managed resource scope that exposes ordinary native tools."""

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        owned = tuple(tools)
        names: set[str] = set()
        for tool in owned:
            if not isinstance(tool, (Tool, AsyncTool)):
                raise TypeError("agents may expose only Tool or AsyncTool instances")
            if tool.name in names:
                raise ValueError(f"duplicate agent tool name: {tool.name}")
            names.add(tool.name)
        self._tools = owned

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """Return the existing native tools exposed by this agent."""

        return self._tools

    @asynccontextmanager
    async def lifespan(self, _ctx: AgentContext) -> AsyncIterator[None]:
        """Optionally acquire resources for this agent's runtime lifetime."""

        yield


@dataclass(slots=True)
class _AgentState:
    name: str
    agent: Agent
    subscriptions: list[tuple[Topic[Any], BoundSubscriber]] = field(
        default_factory=list
    )
    background: set[asyncio.Task[Any]] = field(default_factory=set)
    deliveries: set[asyncio.Task[None]] = field(default_factory=set)
    lifetime: Any = None
    entered: bool = False


class AgentContext:
    """Runtime operations available to an agent lifecycle or subscriber."""

    __slots__ = ("_runtime", "_state", "_metadata")

    def __init__(
        self,
        runtime: AgentRuntime,
        state: _AgentState,
        metadata: MessageMetadata | None,
    ) -> None:
        self._runtime = runtime
        self._state = state
        self._metadata = metadata

    @property
    def agent_name(self) -> str:
        """Return this agent's runtime-local name."""

        return self._state.name

    @property
    def metadata(self) -> MessageMetadata:
        """Return metadata for the current subscription delivery."""

        if self._metadata is None:
            raise RuntimeError("lifecycle context has no message metadata")
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
            source=self._state.name,
            correlation_id=self._metadata.correlation_id if self._metadata else None,
            parent_message_id=self._metadata.message_id if self._metadata else None,
        )

    def start_task(
        self,
        work: Coroutine[Any, Any, TaskResultT],
        *,
        name: str | None = None,
    ) -> asyncio.Task[TaskResultT]:
        """Start background work owned by this agent's runtime lifetime."""

        try:
            self._runtime._ensure_running()
        except RuntimeError:
            work.close()
            raise
        task = asyncio.create_task(work, name=name)
        tracked = cast(asyncio.Task[Any], task)
        self._state.background.add(tracked)
        tracked.add_done_callback(self._background_done)
        return task

    def _resolve_participant(self, participant_id: str | None) -> str:
        if participant_id is not None:
            return participant_id
        if self._metadata is None:
            raise ValueError("participant_id is required outside a subscriber")
        return self._metadata.participant_id

    def _background_done(self, task: asyncio.Task[Any]) -> None:
        self._state.background.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            self._runtime._background_failed(error)


class AgentRuntime:
    """Own agent lifetimes, background tasks, and typed pub/sub routing."""

    def __init__(self) -> None:
        self._agents: dict[str, _AgentState] = {}
        self._topics: dict[str, Topic[Any]] = {}
        self._subscribers: dict[str, list[tuple[_AgentState, BoundSubscriber]]] = {}
        self._running = False
        self._closed = False
        self._failures: list[BaseException] = []

    @property
    def running(self) -> bool:
        """Whether the runtime currently accepts event work."""

        return self._running and not self._closed and not self._failures

    def register(self, name: str, agent: AgentT) -> AgentT:
        """Register one agent before startup and return the same typed object."""

        if self._running or self._closed or self._failures:
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

        state = _AgentState(name=name, agent=agent)
        state.subscriptions = self._discover_subscriptions(agent)
        known_topics = dict(self._topics)
        for topic, _method in state.subscriptions:
            known = known_topics.setdefault(topic.name, topic)
            if known.message_type is not topic.message_type:
                raise ValueError(f"topic {topic.name!r} already uses another message type")

        self._agents[name] = state
        for topic, method in state.subscriptions:
            self._register_topic(topic)
            self._subscribers.setdefault(topic.name, []).append((state, method))
        return agent

    async def start(self) -> None:
        """Enter every registered agent resource scope."""

        if self._closed:
            raise RuntimeClosedError("agent runtime is closed")
        self._raise_if_failed()
        if self._running:
            return
        self._running = True
        try:
            for state in self._agents.values():
                state.lifetime = state.agent.lifespan(AgentContext(self, state, None))
                await state.lifetime.__aenter__()
                state.entered = True
        except (asyncio.CancelledError, Exception):
            await self.stop()
            raise

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
        """Cancel runtime-owned work and then exit every agent scope."""

        if self._closed:
            return
        self._running = False
        tasks = tuple(
            task
            for state in self._agents.values()
            for task in (*state.background, *state.deliveries)
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        errors = list(self._failures)
        for state in reversed(tuple(self._agents.values())):
            if not state.entered:
                continue
            try:
                await state.lifetime.__aexit__(None, None, None)
            except asyncio.CancelledError as exc:
                errors.append(exc)
            except Exception as exc:
                errors.append(exc)
            finally:
                state.entered = False
        self._closed = True
        if errors:
            raise BaseExceptionGroup("errors during agent runtime", errors)

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
        await method(message, AgentContext(self, state, metadata))

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
        self._raise_if_failed()
        if not self.running:
            raise RuntimeClosedError("agent runtime is not running")

    def _raise_if_failed(self) -> None:
        if self._failures:
            raise RuntimeFailedError("agent runtime background work failed") from (
                self._failures[0]
            )

    def _background_failed(self, error: BaseException) -> None:
        self._failures.append(error)
        self._running = False
        for state in self._agents.values():
            for task in (*state.background, *state.deliveries):
                if not task.done():
                    task.cancel()

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
    "Agent",
    "AgentContext",
    "AgentRuntime",
    "MessageMetadata",
    "RuntimeClosedError",
    "RuntimeFailedError",
    "Topic",
    "subscribe",
]
