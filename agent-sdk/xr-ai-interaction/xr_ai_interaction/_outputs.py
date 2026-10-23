# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Producer-side helper that frames :class:`ResponseOutput` messages."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Protocol

from ._messages import ResponseOutput


class _Publisher(Protocol):
    """Anything that publishes one message to a topic for one participant.

    ``AgentRuntime`` and ``RuntimeContext`` both satisfy it.
    """

    async def publish(
        self,
        topic: Any,
        message: Any,
        *,
        participant_id: str | None = ...,
        **kwargs: Any,
    ) -> None: ...


class ResponseOutputs:
    """Publish replies and updates without building protocol messages.

    A producer declares what it emits. The consumer decides when it is
    rendered, so a producer never chooses interruption flags or stream
    identifiers.

    :param publisher: Object with an async ``publish(topic, message, *, participant_id=...)``
        method, such as ``AgentRuntime`` or ``RuntimeContext``.
    :param topic: Topic that carries :class:`ResponseOutput` messages.
    :param source: Producer name passed to ``publish`` as ``source``. Omitted from the call
        when ``None``.
    :param closed_errors: Exception types that mean the consumer is already closed.
        :class:`Reply` suppresses them when it publishes the terminator.
    """

    def __init__(
        self,
        publisher: _Publisher,
        topic: Any,
        *,
        source: str | None = None,
        closed_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._publisher = publisher
        self._topic = topic
        self._source = source
        self._closed_errors = closed_errors

    async def update(
        self,
        participant_id: str,
        text: str,
        *,
        urgent: bool = False,
        timestamp_us: int | None = None,
    ) -> None:
        """Publish unprompted information; blank text publishes nothing.

        An urgent update cuts the output being played and plays next.

        :param participant_id: Participant that hears the update.
        :param text: Update text.
        :param urgent: Whether the update cuts current output and plays next.
        :param timestamp_us: Originating input time as Unix microseconds.
        """

        if not text.strip():
            return
        await self._publish(
            participant_id,
            ResponseOutput(kind="update", text=text, urgent=urgent, timestamp_us=timestamp_us),
        )

    async def withdraw(self, participant_id: str) -> None:
        """Drop this producer's updates for a participant that have not started playing.

        Output already playing and other producers' updates are kept.

        :param participant_id: Participant whose queued updates are dropped.
        """

        await self._publish(participant_id, ResponseOutput(kind="withdraw"))

    def reply(
        self,
        participant_id: str,
        *,
        request_id: str,
        timestamp_us: int | None = None,
        started: Callable[[], Awaitable[None]] | None = None,
    ) -> Reply:
        """Return the context manager for one reply to ``request_id``.

        :param participant_id: Participant that hears the reply.
        :param request_id: Request the reply answers.
        :param timestamp_us: Originating input time as Unix microseconds.
        :param started: Awaited once, before the first fragment is published.
        """

        return Reply(
            self,
            participant_id,
            request_id=request_id,
            timestamp_us=timestamp_us,
            started=started,
        )

    async def _publish(self, participant_id: str, output: ResponseOutput) -> None:
        extra = {} if self._source is None else {"source": self._source}
        await self._publisher.publish(
            self._topic,
            output,
            participant_id=participant_id,
            **extra,
        )


class Reply:
    """Async context manager that publishes one reply, streamed or single.

    Create it with :meth:`ResponseOutputs.reply`. Every :meth:`send` publishes
    one fragment of the reply, and leaving the context publishes the empty
    terminator that completes it. A single-message reply is one ``send``.

    - Fragments that are blank before the first published fragment are skipped.
    - A canceled task never publishes: ``send`` raises
      :class:`asyncio.CancelledError` when the current task is being canceled,
      even if the code it awaited swallowed the cancellation.
    - On exit, whether normal, by exception, or by cancellation, the terminator
      is published if any fragment was published and no publish failed. A
      failed publish is never retried or closed. Exceptions listed in the
      ``closed_errors`` of the :class:`ResponseOutputs` are suppressed when
      they come from the terminator.
    - Calling ``send`` after the context exits raises :class:`RuntimeError`.
    """

    def __init__(
        self,
        outputs: ResponseOutputs,
        participant_id: str,
        *,
        request_id: str,
        timestamp_us: int | None = None,
        started: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._outputs = outputs
        self._participant_id = participant_id
        self._request_id = request_id
        self._timestamp_us = timestamp_us
        self._started = started
        self._response_id = uuid.uuid4().hex
        self._published = False
        self._failed = False
        self._exited = False
        self._delivered = False

    @property
    def delivered(self) -> bool:
        """Whether the terminator was published; meaningful after the context exits."""

        return self._delivered

    async def __aenter__(self) -> Reply:
        return self

    async def send(self, text: str) -> None:
        """Publish one fragment of the reply."""

        if self._exited:
            raise RuntimeError("reply is already closed")
        if not self._published and not text.strip():
            return
        self._raise_if_cancelling()
        if not self._published and self._started is not None:
            await self._started()
            self._raise_if_cancelling()
        try:
            await self._outputs._publish(
                self._participant_id,
                ResponseOutput(
                    kind="reply",
                    text=text,
                    request_id=self._request_id,
                    response_id=self._response_id,
                    final=False,
                    timestamp_us=self._timestamp_us,
                ),
            )
        except Exception:
            self._failed = True
            raise
        self._published = True

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._exited = True
        if not self._published or self._failed:
            return
        try:
            await self._outputs._publish(
                self._participant_id,
                ResponseOutput(
                    kind="reply",
                    request_id=self._request_id,
                    response_id=self._response_id,
                    final=True,
                    timestamp_us=self._timestamp_us,
                ),
            )
        except self._outputs._closed_errors:
            return
        self._delivered = True

    @staticmethod
    def _raise_if_cancelling() -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
