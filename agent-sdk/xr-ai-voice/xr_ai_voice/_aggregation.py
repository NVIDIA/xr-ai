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

from ._runtime import VOICE_OUTPUT_TOPIC, VoiceOutput, VoicePriority

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
    pending: deque[_Contribution] = field(default_factory=deque)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    discarded_streams: dict[tuple[str, str], float] = field(default_factory=dict)
    in_flight_count: int = 0
    task: asyncio.Task[None] | None = None


class VoiceAggregationAgent(Agent):
    """Serialize and coalesce candidate speech independently per participant.

    A lone finite contribution passes through without a model call after the
    short coalescing window. Completed text is finalized downstream immediately,
    while the aggregator independently reserves its estimated spoken duration
    so later updates wait and coalesce instead of building a speech queue.
    Multiple pending finite contributions are rewritten into one response. A
    lone incremental response streams through immediately and uses the same
    scheduling reservation after its final chunk. High-priority contributions
    move ahead of routine pending work without interrupting active output. An
    urgent contribution interrupts the active output.
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
        speech_rate_wpm: float = 150.0,
        minimum_playback_s: float = 0.4,
        maximum_playback_s: float = 30.0,
        rewrite_timeout_s: float = 5.0,
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
        if speech_rate_wpm <= 0:
            raise ValueError("voice aggregation speech rate must be positive")
        if minimum_playback_s < 0:
            raise ValueError("voice aggregation minimum playback must not be negative")
        if maximum_playback_s <= 0:
            raise ValueError("voice aggregation maximum playback must be positive")
        if minimum_playback_s > maximum_playback_s:
            raise ValueError("voice aggregation minimum playback must not exceed maximum playback")
        if rewrite_timeout_s <= 0:
            raise ValueError("voice aggregation rewrite timeout must be positive")
        super().__init__()
        self._llm = llm
        self._prompt = prompt.strip()
        self._coalesce_window_s = coalesce_window_s
        self._queue_capacity = queue_capacity
        self._max_batch_size = max_batch_size
        self._max_tokens = max_tokens
        self._stream_idle_timeout_s = stream_idle_timeout_s
        self._participant_idle_timeout_s = participant_idle_timeout_s
        self._speech_rate_wpm = speech_rate_wpm
        self._minimum_playback_s = minimum_playback_s
        self._maximum_playback_s = maximum_playback_s
        self._rewrite_timeout_s = rewrite_timeout_s
        self._states: dict[str, _ParticipantState] = {}
        self._stopping = False

    @subscribe(VOICE_CONTRIBUTION_TOPIC)
    async def contribute(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        """Enqueue one finite response or incremental response fragment."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("voice contribution requires a participant")
        if self._stopping:
            logger.debug(
                "dropped voice contribution while stopping pid={!r} source={!r}",
                participant_id,
                ctx.metadata.source,
            )
            return
        state = self._states.get(participant_id)
        if state is None or state.task is None or state.task.done():
            state = _ParticipantState()
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
        self._enqueue(
            state,
            _Contribution(
                output=output,
                ctx=ctx,
                participant_id=participant_id,
                source=ctx.metadata.source,
            ),
        )

    async def stop(self) -> None:
        """Cancel and await all participant aggregation tasks."""

        self._stopping = True
        tasks = tuple(
            (participant_id, state, state.task)
            for participant_id, state in self._states.items()
            if state.task is not None
        )
        self._states.clear()
        for participant_id, state, task in tasks:
            self._log_discarded(participant_id, state, reason="shutdown")
            task.cancel()
        if tasks:
            await asyncio.gather(
                *(task for _participant_id, _state, task in tasks),
                return_exceptions=True,
            )

    async def release(self, participant_id: str) -> None:
        """Cancel and release one departed participant's aggregation state."""

        state = self._states.pop(participant_id, None)
        if state is None or state.task is None:
            return
        self._log_discarded(participant_id, state, reason="participant release")
        state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)

    async def _run_participant(
        self,
        participant_id: str,
        state: _ParticipantState,
    ) -> None:
        while True:
            try:
                contribution = await asyncio.wait_for(
                    self._wait_for_next(state),
                    timeout=self._participant_idle_timeout_s,
                )
            except TimeoutError:
                contribution = self._pop_next(state)
                if contribution is None:
                    return
            await self._process_contribution(participant_id, state, contribution)

    async def _process_contribution(
        self,
        participant_id: str,
        state: _ParticipantState,
        contribution: _Contribution,
    ) -> None:
        self._prune_expired_streams(
            state,
            asyncio.get_running_loop().time(),
        )
        state.in_flight_count = 1
        key = contribution.stream_key
        try:
            if key is not None and self._discarded_stream_fragment(
                state,
                key,
                final=contribution.output.final,
            ):
                return
            if key is not None and not contribution.output.final:
                await self._forward_stream(participant_id, state, contribution)
                return
            batch = await self._finite_batch(state, contribution)
            if batch:
                await self._speak_batch(participant_id, state, batch)
        finally:
            state.in_flight_count = 0

    def _enqueue(
        self,
        state: _ParticipantState,
        contribution: _Contribution,
    ) -> None:
        if len(state.pending) >= self._queue_capacity:
            incoming_rank = self._schedule_rank(contribution)
            candidates = [
                (index, self._schedule_rank(pending))
                for index, pending in enumerate(state.pending)
                if contribution.output.interrupt
                or (
                    not pending.output.interrupt
                    and self._schedule_rank(pending) <= incoming_rank
                )
            ]
            if not candidates:
                self._drop_contribution(state, contribution, incoming=True)
                return
            lowest_rank = min(rank for _index, rank in candidates)
            victim_index = next(
                index for index, rank in candidates if rank == lowest_rank
            )
            victim = state.pending[victim_index]
            del state.pending[victim_index]
            self._drop_contribution(state, victim, incoming=False)
        state.pending.append(contribution)
        state.changed.set()

    def _drop_contribution(
        self,
        state: _ParticipantState,
        contribution: _Contribution,
        *,
        incoming: bool,
    ) -> None:
        key = contribution.stream_key
        if key is not None:
            self._discard_stream(state, key)
        logger.warning(
            "dropped {} voice contribution pid={!r} source={!r} response_id={!r} interrupt={} capacity={}",
            "incoming" if incoming else "oldest pending",
            contribution.participant_id,
            contribution.source,
            contribution.output.response_id,
            contribution.output.interrupt,
            self._queue_capacity,
        )

    def _restore_batch(
        self,
        state: _ParticipantState,
        batch: list[_Contribution],
    ) -> None:
        state.pending.extendleft(reversed(batch))
        while len(state.pending) > self._queue_capacity:
            victim_index = min(
                range(len(state.pending)),
                key=lambda index: self._schedule_rank(state.pending[index]),
            )
            victim = state.pending[victim_index]
            del state.pending[victim_index]
            self._drop_contribution(state, victim, incoming=False)
        state.changed.set()

    def _discard_stream(
        self,
        state: _ParticipantState,
        stream_key: tuple[str, str],
    ) -> None:
        now = asyncio.get_running_loop().time()
        self._prune_expired_streams(state, now)
        state.discarded_streams[stream_key] = now + self._stream_idle_timeout_s
        state.changed.set()

    @staticmethod
    def _prune_expired_streams(
        state: _ParticipantState,
        now: float,
    ) -> None:
        for key, expires_at in tuple(state.discarded_streams.items()):
            if expires_at <= now:
                state.discarded_streams.pop(key, None)

    def _discarded_stream_fragment(
        self,
        state: _ParticipantState,
        stream_key: tuple[str, str],
        *,
        final: bool,
    ) -> bool:
        now = asyncio.get_running_loop().time()
        expires_at = state.discarded_streams.get(stream_key)
        if expires_at is None:
            return False
        if expires_at <= now:
            state.discarded_streams.pop(stream_key, None)
            return False
        if final:
            state.discarded_streams.pop(stream_key, None)
        else:
            state.discarded_streams[stream_key] = now + self._stream_idle_timeout_s
        return True

    def _stream_is_discarded(
        self,
        state: _ParticipantState,
        stream_key: tuple[str, str],
    ) -> bool:
        expires_at = state.discarded_streams.get(stream_key)
        if expires_at is None:
            return False
        if expires_at <= asyncio.get_running_loop().time():
            state.discarded_streams.pop(stream_key, None)
            return False
        return True

    @classmethod
    def _pop_next(cls, state: _ParticipantState) -> _Contribution | None:
        if not state.pending:
            return None
        index = max(
            range(len(state.pending)),
            key=lambda item: cls._schedule_rank(state.pending[item]),
        )
        contribution = state.pending[index]
        del state.pending[index]
        return contribution

    async def _wait_for_next(self, state: _ParticipantState) -> _Contribution:
        while True:
            contribution = self._pop_next(state)
            if contribution is not None:
                return contribution
            state.changed.clear()
            contribution = self._pop_next(state)
            if contribution is not None:
                return contribution
            await state.changed.wait()

    async def _finite_batch(
        self,
        state: _ParticipantState,
        first: _Contribution,
    ) -> list[_Contribution]:
        if first.output.interrupt:
            return [first]
        if self._coalesce_window_s:
            await asyncio.sleep(self._coalesce_window_s)
        batch = [first]
        priority = first.output.priority
        while len(batch) < self._max_batch_size:
            contribution = self._pop_next(state)
            if contribution is None:
                break
            key = contribution.stream_key
            if key is not None and self._discarded_stream_fragment(
                state,
                key,
                final=contribution.output.final,
            ):
                continue
            if key is not None and not contribution.output.final:
                state.pending.appendleft(contribution)
                preempting = self._pop_finite_preemption(state, priority)
                if preempting is not None:
                    self._restore_batch(state, batch)
                    if preempting.output.interrupt:
                        return [preempting]
                    state.pending.appendleft(preempting)
                    state.changed.set()
                    return []
                state.changed.set()
                break
            if contribution.output.interrupt:
                state.pending.extendleft(reversed(batch))
                state.changed.set()
                return [contribution]
            if self._priority_rank(contribution.output.priority) > self._priority_rank(
                priority
            ):
                self._restore_batch(state, batch)
                state.pending.appendleft(contribution)
                state.changed.set()
                return []
            if contribution.output.priority != priority:
                state.pending.appendleft(contribution)
                state.changed.set()
                break
            batch.append(contribution)
        return batch

    async def _forward_stream(
        self,
        participant_id: str,
        state: _ParticipantState,
        first: _Contribution,
    ) -> None:
        input_key = first.stream_key
        assert input_key is not None
        output_id = uuid.uuid4().hex
        started_at = asyncio.get_running_loop().time()
        spoken_text = first.output.text
        try:
            await self._publish_stream(first, output_id, interrupt=first.output.interrupt)
            state.in_flight_count = 0

            while True:
                try:
                    contribution = await asyncio.wait_for(
                        self._wait_for_stream_contribution(state, input_key),
                        timeout=self._stream_idle_timeout_s,
                    )
                except TimeoutError:
                    logger.warning(
                        "voice contribution stream timed out pid={!r} source={!r} response_id={!r}",
                        participant_id,
                        input_key[0],
                        input_key[1],
                    )
                    self._discard_stream(state, input_key)
                    await self._publish_stream_end(first, output_id)
                    return

                if contribution is None:
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
                    spoken_text += contribution.output.text
                    state.in_flight_count = 1
                    # Downstream finality follows content availability. The
                    # playback reservation below is only scheduler state and
                    # must not delay the client's completed-text echo.
                    await self._publish_stream(
                        contribution,
                        output_id,
                        interrupt=False,
                    )
                    state.in_flight_count = 0
                    if contribution.output.final:
                        elapsed = asyncio.get_running_loop().time() - started_at
                        remaining = max(
                            0.0,
                            self._playback_duration(spoken_text) - elapsed,
                        )
                        urgent = await self._hold_output(state, remaining)
                        if urgent is not None:
                            await self._dispatch_urgent(
                                participant_id,
                                state,
                                urgent,
                            )
                            return
                        return
                    continue

                self._discard_stream(state, input_key)
                state.in_flight_count = 1
                await self._dispatch_urgent(participant_id, state, contribution)
                return
        finally:
            state.in_flight_count = 0

    async def _wait_for_stream_contribution(
        self,
        state: _ParticipantState,
        input_key: tuple[str, str],
    ) -> _Contribution | None:
        while True:
            if self._stream_is_discarded(state, input_key):
                return None
            contribution = self._pop_stream_contribution(state, input_key)
            if contribution is not None:
                return contribution
            state.changed.clear()
            if self._stream_is_discarded(state, input_key):
                return None
            contribution = self._pop_stream_contribution(state, input_key)
            if contribution is not None:
                return contribution
            await state.changed.wait()

    @staticmethod
    def _pop_stream_contribution(
        state: _ParticipantState,
        input_key: tuple[str, str],
    ) -> _Contribution | None:
        for index, contribution in enumerate(state.pending):
            if contribution.output.interrupt or contribution.stream_key == input_key:
                del state.pending[index]
                return contribution
        return None

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
                    priority=contribution.output.priority,
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
                    priority=contribution.output.priority,
                    timestamp_us=contribution.output.timestamp_us,
                ),
            )
        except RuntimeClosedError:
            return

    async def _speak_batch(
        self,
        participant_id: str,
        state: _ParticipantState,
        batch: list[_Contribution],
    ) -> None:
        state.in_flight_count = len(batch)
        interrupts = any(contribution.output.interrupt for contribution in batch)
        if len(batch) == 1:
            text = batch[0].output.text.strip()
        elif interrupts:
            text = self._fallback_text(batch)
        else:
            rewrite = asyncio.create_task(
                self._rewrite(participant_id, batch),
                name=f"voice-aggregation-rewrite:{participant_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
            preempting = asyncio.create_task(
                self._wait_for_preemption(state, batch[0].output.priority),
                name=f"voice-aggregation-preemption:{participant_id}",
            )
            try:
                done, _pending = await asyncio.wait(
                    (rewrite, preempting),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if preempting in done:
                    state.in_flight_count += 1
                    logger.debug(
                        "higher-priority voice contribution preempted rewrite pid={!r}",
                        participant_id,
                    )
                    rewrite.cancel()
                    await asyncio.gather(rewrite, return_exceptions=True)
                    self._restore_batch(state, batch)
                    state.in_flight_count = 1
                    await self._dispatch_preempting(
                        participant_id,
                        state,
                        preempting.result(),
                    )
                    return
                text = rewrite.result()
            finally:
                for task in (rewrite, preempting):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(rewrite, preempting, return_exceptions=True)
        if not text:
            state.in_flight_count = 0
            return
        timestamp_us = min(
            (
                contribution.output.timestamp_us
                for contribution in batch
                if contribution.output.timestamp_us is not None
            ),
            default=None,
        )
        output_id = uuid.uuid4().hex
        output = _Contribution(
            output=VoiceOutput(
                text=text,
                response_id=output_id,
                # The rewrite/raw batch is complete before playback pacing.
                # Closing it here lets the data echo reach the client now.
                final=True,
                interrupt=interrupts,
                priority=batch[0].output.priority,
                timestamp_us=timestamp_us,
            ),
            ctx=batch[0].ctx,
            participant_id=batch[0].participant_id,
            source=batch[0].source,
        )
        await self._publish_stream(
            output,
            output_id,
            interrupt=output.output.interrupt,
        )
        state.in_flight_count = 0
        urgent_contribution = await self._hold_output(
            state,
            self._playback_duration(text),
        )
        if urgent_contribution is not None:
            await self._dispatch_urgent(
                participant_id,
                state,
                urgent_contribution,
            )
            return
        priority_contribution = await self._hold_priority_delay(
            state,
            self._post_playback_delay(output.output),
        )
        if priority_contribution is not None:
            await self._dispatch_preempting(
                participant_id,
                state,
                priority_contribution,
            )

    async def _hold_output(
        self,
        state: _ParticipantState,
        duration_s: float,
    ) -> _Contribution | None:
        """Wait for playback pacing to expire or urgent output to supersede it."""

        if duration_s <= 0:
            return None
        try:
            return await asyncio.wait_for(
                self._wait_for_interrupt(state),
                timeout=duration_s,
            )
        except TimeoutError:
            return None

    async def _wait_for_interrupt(
        self,
        state: _ParticipantState,
    ) -> _Contribution:
        while True:
            contribution = self._pop_interrupt(state)
            if contribution is not None:
                return contribution
            state.changed.clear()
            contribution = self._pop_interrupt(state)
            if contribution is not None:
                return contribution
            await state.changed.wait()

    async def _hold_priority_delay(
        self,
        state: _ParticipantState,
        duration_s: float,
    ) -> _Contribution | None:
        if duration_s <= 0:
            return None
        try:
            return await asyncio.wait_for(
                self._wait_for_preemption(state, VoicePriority.NORMAL),
                timeout=duration_s,
            )
        except TimeoutError:
            return None

    async def _wait_for_preemption(
        self,
        state: _ParticipantState,
        priority: VoicePriority,
    ) -> _Contribution:
        while True:
            contribution = self._pop_preemption(state, priority)
            if contribution is not None:
                return contribution
            state.changed.clear()
            contribution = self._pop_preemption(state, priority)
            if contribution is not None:
                return contribution
            await state.changed.wait()

    @staticmethod
    def _pop_interrupt(state: _ParticipantState) -> _Contribution | None:
        for index, contribution in enumerate(state.pending):
            if contribution.output.interrupt:
                del state.pending[index]
                return contribution
        return None

    @classmethod
    def _pop_preemption(
        cls,
        state: _ParticipantState,
        priority: VoicePriority,
    ) -> _Contribution | None:
        baseline = cls._priority_rank(priority)
        candidates = [
            index
            for index, contribution in enumerate(state.pending)
            if contribution.output.interrupt
            or cls._priority_rank(contribution.output.priority) > baseline
        ]
        if not candidates:
            return None
        index = max(
            candidates,
            key=lambda item: cls._schedule_rank(state.pending[item]),
        )
        contribution = state.pending[index]
        del state.pending[index]
        return contribution

    @classmethod
    def _pop_finite_preemption(
        cls,
        state: _ParticipantState,
        priority: VoicePriority,
    ) -> _Contribution | None:
        baseline = cls._priority_rank(priority)
        candidates = [
            index
            for index, contribution in enumerate(state.pending)
            if (
                contribution.output.interrupt
                or cls._priority_rank(contribution.output.priority) > baseline
            )
            and (contribution.stream_key is None or contribution.output.final)
        ]
        if not candidates:
            return None
        index = max(
            candidates,
            key=lambda item: cls._schedule_rank(state.pending[item]),
        )
        contribution = state.pending[index]
        del state.pending[index]
        return contribution

    async def _dispatch_urgent(
        self,
        participant_id: str,
        state: _ParticipantState,
        contribution: _Contribution,
    ) -> None:
        state.in_flight_count = 1
        key = contribution.stream_key
        if key is not None and not contribution.output.final:
            await self._forward_stream(participant_id, state, contribution)
            return
        await self._speak_batch(participant_id, state, [contribution])

    async def _dispatch_preempting(
        self,
        participant_id: str,
        state: _ParticipantState,
        contribution: _Contribution,
    ) -> None:
        if contribution.output.interrupt:
            await self._dispatch_urgent(participant_id, state, contribution)
            return
        await self._process_contribution(participant_id, state, contribution)

    def _playback_duration(self, text: str) -> float:
        words = len(text.split())
        estimated_s = words * 60.0 / self._speech_rate_wpm
        return min(
            self._maximum_playback_s,
            max(self._minimum_playback_s, estimated_s),
        )

    def _post_playback_delay(self, _output: VoiceOutput) -> float:
        return 0.0

    @staticmethod
    def _priority_rank(priority: VoicePriority) -> int:
        return 1 if priority is VoicePriority.HIGH else 0

    @classmethod
    def _schedule_rank(cls, contribution: _Contribution) -> tuple[int, int]:
        return (
            int(contribution.output.interrupt),
            cls._priority_rank(contribution.output.priority),
        )

    @staticmethod
    def _fallback_text(batch: list[_Contribution]) -> str:
        return " ".join(contribution.output.text.strip() for contribution in batch if contribution.output.text.strip())

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
        fallback = self._fallback_text(batch)
        if len(updates) < 2:
            return fallback
        try:
            async with asyncio.timeout(self._rewrite_timeout_s):
                response = await self._llm.chat(
                    (
                        ChatMessage(role="system", content=self._prompt),
                        ChatMessage(
                            role="user",
                            content="Pending spoken updates:\n" + json.dumps(updates, ensure_ascii=False),
                        ),
                    ),
                    max_tokens=self._max_tokens,
                    temperature=0.0,
                    enable_thinking=False,
                    timeout=self._rewrite_timeout_s,
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

    @staticmethod
    def _log_discarded(
        participant_id: str,
        state: _ParticipantState,
        *,
        reason: str,
    ) -> None:
        count = len(state.pending) + state.in_flight_count
        if count:
            logger.warning(
                "discarded accepted voice contributions pid={!r} count={} reason={}",
                participant_id,
                count,
                reason,
            )
        else:
            logger.debug(
                "released voice aggregation state pid={!r} reason={}",
                participant_id,
                reason,
            )

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
