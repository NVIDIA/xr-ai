# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response framing owned by xr_ai_voice; agents own production and cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import TracebackType
from typing import Literal

from xr_ai_runtime import AgentRuntime, Topic

from ._runtime import VoiceOutput


class VoiceOutputPublisher:
    """Publish one agent-owned response directly to a configured runtime topic.

    Use once in ``async with`` in the task that owns the response. ``send``
    awaits delivery, preserving downstream backpressure. The first nonblank
    body carries ``interrupt``; all frames share the supplied participant,
    source, response identifier, and originating input timestamp. Identifiers
    must be unique per response for that participant and source.

    An opened incremental response ends with one empty final frame when the
    context exits normally or with a producer exception. Cancellation and
    publication failure prevent further frames. All failures propagate without
    retry; delivery may have partially succeeded, so a failed publisher cannot
    be reused for fallback. Applications own fallback text, model iterators,
    their cleanup, and response task cancellation. No task is created here.

    ``on_first_body``, when supplied, runs synchronously in the owning task
    immediately before the first body publication attempt. It is never called
    for empty output. Its failure propagates and prevents publication.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        output_topic: Topic[VoiceOutput],
        *,
        participant_id: str,
        source: str,
        response_id: str,
        timestamp_us: int | None = None,
        interrupt: bool = True,
        on_first_body: Callable[[], object] | None = None,
    ) -> None:
        if not participant_id.strip() or not source.strip() or not response_id.strip():
            raise ValueError("participant, source, and response identifiers must not be blank")
        if output_topic.message_type is not VoiceOutput:
            raise TypeError("output topic must carry VoiceOutput")
        if timestamp_us is not None and timestamp_us < 0:
            raise ValueError("timestamp_us must not be negative")
        self._runtime = runtime
        self._topic = output_topic
        self._participant_id = participant_id
        self._source = source
        self._response_id = response_id
        self._timestamp_us = timestamp_us
        self._interrupt = interrupt
        self._on_first_body = on_first_body
        self._started = False
        self._state: Literal["new", "active", "final", "failed", "closed"] = "new"
        self._owner: asyncio.Task[object] | None = None

    @property
    def started(self) -> bool:
        """Whether the first body publication completed successfully.

        This can guide application fallback after a producer failure. A failed
        publication still prevents further sends, even when this is false.
        """

        return self._started

    async def __aenter__(self) -> VoiceOutputPublisher:
        """Start this response in its owning task; a publisher is single-use."""

        if self._state != "new":
            raise ValueError("voice output publisher is single-use")
        self._owner = asyncio.current_task()
        self._state = "active"
        self._check_owner()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close partial output unless cancelled or publication failed."""

        try:
            self._check_owner()
            if exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
                return
            if self._state == "active" and self._started:
                await self._publish("", final=True)
        finally:
            self._state = "closed"

    async def send(self, text: str, *, final: bool = False) -> None:
        """Await one body fragment, or publish a finite response with ``final=True``.

        Leading whitespace-only fragments are ignored. Once output starts,
        whitespace fragments are preserved. An empty final body closes an open
        stream; an empty response publishes nothing. Sending after a final
        body or failed publication is a caller error.

        Subscriber failures propagate unchanged as runtime publication exception
        groups. Publication failure prevents fallback through this publisher.

        Raises:
            xr_ai_runtime.RuntimeClosedError: The runtime no longer accepts output.
        """

        self._check_owner()
        if self._state != "active":
            raise ValueError("voice output publisher is not accepting fragments")
        if not text and not final:
            return
        if not self._started and not text.strip():
            if final:
                self._state = "final"
            return
        await self._publish(text, final=final)

    def _check_owner(self) -> None:
        task = asyncio.current_task()
        if task is None or task is not self._owner:
            raise ValueError("voice output must be published by its owning task")
        # A model iterator can swallow cancellation without clearing the request.
        if task.cancelling():
            raise asyncio.CancelledError

    async def _publish(self, text: str, *, final: bool) -> None:
        output = VoiceOutput(
            text=text,
            response_id=self._response_id,
            final=final,
            interrupt=self._interrupt and not self._started,
            timestamp_us=self._timestamp_us,
        )
        # Delivery can partially succeed before raising; never retry or finalize it.
        self._state = "failed"
        if not self._started and self._on_first_body is not None:
            self._on_first_body()
        self._check_owner()
        await self._runtime.publish(
            self._topic,
            output,
            participant_id=self._participant_id,
            source=self._source,
        )
        self._started = True
        self._state = "final" if final else "active"
        self._check_owner()
