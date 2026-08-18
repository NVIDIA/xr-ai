# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped aggregation for concurrent voice producers."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass, field

import nemo_relay
from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, Topic, subscribe

from ._runtime import VOICE_OUTPUT_TOPIC, VoiceOutput

VOICE_CONTRIBUTION_TOPIC = Topic(
    "voice.contribution",
    VoiceOutput,
    telemetry="none",
)
"""Candidate spoken output consumed by :class:`VoiceAggregationAgent`."""

_DEFAULT_PROMPT = """Combine simultaneous spoken updates into one concise, natural response.
Preserve every actionable fact, proper name, value, unit, and warning. Remove repetition.
Do not invent information or mention agents, queues, sources, or aggregation.
Use plain spoken prose with at most two sentences."""


@dataclass(frozen=True, slots=True)
class _Contribution:
    output: VoiceOutput
    ctx: RuntimeContext
    participant_id: str
    source: str

    @property
    def stream_key(self) -> tuple[str, str] | None:
        if self.output.response_id is None:
            return None
        return (self.source, self.output.response_id)


@dataclass(slots=True)
class _ParticipantState:
    queue: asyncio.Queue[_Contribution]
    backlog: deque[_Contribution] = field(default_factory=deque)
    discarded_streams: set[tuple[str, str]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None


class VoiceAggregationAgent(Agent):
    """Serialize and coalesce candidate speech independently per participant.

    A lone finite contribution passes through without a model call after the
    short coalescing window. Multiple finite contributions in that window are
    rewritten into one response. A lone incremental response streams through
    immediately; other contributions wait until that stream ends. An urgent
    contribution interrupts the active stream and is combined with pending
    finite updates.
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        prompt: str = _DEFAULT_PROMPT,
        coalesce_window_s: float = 0.15,
        queue_capacity: int = 64,
        max_batch_size: int = 8,
        max_tokens: int = 192,
        stream_idle_timeout_s: float = 15.0,
        participant_idle_timeout_s: float = 60.0,
    ) -> None:
        if not prompt.strip():
            raise ValueError("voice aggregation prompt must not be empty")
        if coalesce_window_s < 0:
            raise ValueError("voice aggregation window must not be negative")
        if queue_capacity <= 0:
            raise ValueError("voice aggregation queue capacity must be positive")
        if max_batch_size < 2:
            raise ValueError("voice aggregation batch size must be at least two")
        if max_tokens <= 0:
            raise ValueError("voice aggregation token limit must be positive")
        if stream_idle_timeout_s <= 0:
            raise ValueError("voice stream idle timeout must be positive")
        if participant_idle_timeout_s <= 0:
            raise ValueError("voice participant idle timeout must be positive")
        super().__init__()
        self._llm = llm
        self._prompt = prompt.strip()
        self._coalesce_window_s = coalesce_window_s
        self._queue_capacity = queue_capacity
        self._max_batch_size = max_batch_size
        self._max_tokens = max_tokens
        self._stream_idle_timeout_s = stream_idle_timeout_s
        self._participant_idle_timeout_s = participant_idle_timeout_s
        self._states: dict[str, _ParticipantState] = {}
        self._stopping = False

    @subscribe(VOICE_CONTRIBUTION_TOPIC)
    async def contribute(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        """Enqueue one finite response or incremental response fragment."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("voice contribution requires a participant")
        if self._stopping:
            raise RuntimeError("voice aggregation agent is stopping")
        state = self._states.get(participant_id)
        if state is None or state.task is None or state.task.done():
            state = _ParticipantState(
                queue=asyncio.Queue(maxsize=self._queue_capacity),
            )
            task = asyncio.create_task(
                self._run_participant(participant_id, state),
                name=f"voice-aggregation:{participant_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
            state.task = task
            self._states[participant_id] = state
            task.add_done_callback(
                lambda completed, pid=participant_id, owned=state: self._discard(
                    pid,
                    owned,
                    completed,
                )
            )
        await state.queue.put(
            _Contribution(
                output=output,
                ctx=ctx,
                participant_id=participant_id,
                source=ctx.metadata.source,
            )
        )

    async def stop(self) -> None:
        """Cancel and await all participant aggregation tasks."""

        self._stopping = True
        tasks = tuple(
            state.task
            for state in self._states.values()
            if state.task is not None
        )
        self._states.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_participant(
        self,
        participant_id: str,
        state: _ParticipantState,
    ) -> None:
        while True:
            try:
                contribution = await asyncio.wait_for(
                    self._next(state),
                    timeout=self._participant_idle_timeout_s,
                )
            except TimeoutError:
                return
            key = contribution.stream_key
            if key is not None and key in state.discarded_streams:
                if contribution.output.final:
                    state.discarded_streams.discard(key)
                continue
            if key is not None and not contribution.output.final:
                await self._forward_stream(participant_id, state, contribution)
                continue
            batch = await self._finite_batch(state, contribution)
            await self._speak_batch(participant_id, batch)

    async def _next(self, state: _ParticipantState) -> _Contribution:
        if state.backlog:
            return state.backlog.popleft()
        return await state.queue.get()

    async def _finite_batch(
        self,
        state: _ParticipantState,
        first: _Contribution,
    ) -> list[_Contribution]:
        if self._coalesce_window_s:
            await asyncio.sleep(self._coalesce_window_s)
        batch = [first]
        while len(batch) < self._max_batch_size:
            contribution = self._next_nowait(state)
            if contribution is None:
                break
            key = contribution.stream_key
            if key is not None and key in state.discarded_streams:
                if contribution.output.final:
                    state.discarded_streams.discard(key)
                continue
            if key is not None and not contribution.output.final:
                state.backlog.appendleft(contribution)
                break
            batch.append(contribution)
        return batch

    @staticmethod
    def _next_nowait(state: _ParticipantState) -> _Contribution | None:
        if state.backlog:
            return state.backlog.popleft()
        try:
            return state.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _forward_stream(
        self,
        participant_id: str,
        state: _ParticipantState,
        first: _Contribution,
    ) -> None:
        input_key = first.stream_key
        assert input_key is not None
        output_id = uuid.uuid4().hex
        await self._publish_stream(first, output_id, interrupt=first.output.interrupt)

        while True:
            try:
                contribution = await asyncio.wait_for(
                    state.queue.get(),
                    timeout=self._stream_idle_timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "voice contribution stream timed out pid={!r} source={!r} response_id={!r}",
                    participant_id,
                    input_key[0],
                    input_key[1],
                )
                state.discarded_streams.add(input_key)
                await self._publish_stream_end(first, output_id)
                return

            if contribution.stream_key == input_key:
                if contribution.output.interrupt:
                    logger.warning(
                        "ignored interrupt on non-initial voice contribution chunk "
                        "pid={!r} source={!r} response_id={!r}",
                        participant_id,
                        input_key[0],
                        input_key[1],
                    )
                await self._publish_stream(contribution, output_id, interrupt=False)
                if contribution.output.final:
                    return
                continue

            if contribution.output.interrupt:
                state.discarded_streams.add(input_key)
                state.backlog.appendleft(contribution)
                return
            state.backlog.append(contribution)

    async def _publish_stream(
        self,
        contribution: _Contribution,
        response_id: str,
        *,
        interrupt: bool,
    ) -> None:
        try:
            await contribution.ctx.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(
                    text=contribution.output.text,
                    response_id=response_id,
                    final=contribution.output.final,
                    interrupt=interrupt,
                    timestamp_us=contribution.output.timestamp_us,
                ),
            )
        except RuntimeClosedError:
            return

    async def _publish_stream_end(
        self,
        contribution: _Contribution,
        response_id: str,
    ) -> None:
        try:
            await contribution.ctx.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(
                    response_id=response_id,
                    timestamp_us=contribution.output.timestamp_us,
                ),
            )
        except RuntimeClosedError:
            return

    async def _speak_batch(
        self,
        participant_id: str,
        batch: list[_Contribution],
    ) -> None:
        text = (
            batch[0].output.text.strip()
            if len(batch) == 1
            else await self._rewrite(participant_id, batch)
        )
        if not text:
            return
        timestamp_us = min(
            (
                contribution.output.timestamp_us
                for contribution in batch
                if contribution.output.timestamp_us is not None
            ),
            default=None,
        )
        try:
            await batch[0].ctx.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(
                    text=text,
                    interrupt=any(
                        contribution.output.interrupt
                        for contribution in batch
                    ),
                    timestamp_us=timestamp_us,
                ),
            )
        except RuntimeClosedError:
            return

    async def _rewrite(
        self,
        participant_id: str,
        batch: list[_Contribution],
    ) -> str:
        updates = [
            {
                "source": contribution.source,
                "text": contribution.output.text.strip(),
            }
            for contribution in batch
            if contribution.output.text.strip()
        ]
        fallback = " ".join(update["text"] for update in updates)
        if len(updates) < 2:
            return fallback
        try:
            response = await self._llm.chat(
                (
                    ChatMessage(role="system", content=self._prompt),
                    ChatMessage(
                        role="user",
                        content="Pending spoken updates:\n"
                        + json.dumps(updates, ensure_ascii=False),
                    ),
                ),
                max_tokens=self._max_tokens,
                temperature=0.0,
                enable_thinking=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).error(
                "voice contribution rewrite failed pid={!r}",
                participant_id,
            )
            return fallback
        text = response.content.strip()
        if not text or response.tool_calls:
            logger.warning(
                "voice contribution rewrite returned no usable text pid={!r}",
                participant_id,
            )
            return fallback
        return text

    def _discard(
        self,
        participant_id: str,
        state: _ParticipantState,
        task: asyncio.Task[None],
    ) -> None:
        if self._states.get(participant_id) is state:
            self._states.pop(participant_id, None)
        if task.cancelled():
            return
        if error := task.exception():
            logger.error(
                "voice aggregation stopped pid={!r}: {!r}",
                participant_id,
                error,
            )


__all__ = ["VOICE_CONTRIBUTION_TOPIC", "VoiceAggregationAgent"]
