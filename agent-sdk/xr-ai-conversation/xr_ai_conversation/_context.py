# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private participant dialogue and bounded source-context storage.

Only modules in ``xr_ai_conversation`` use this implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import nemo_relay


def _format_minute(timestamp_us: int) -> str:
    """Format an exchange timestamp at the conversation's display precision."""

    return datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).strftime("%Y-%m-%d %H:%M UTC")


@dataclass(frozen=True, slots=True)
class _TranscriptEntry:
    sequence: int
    source: str
    user: str
    assistant: str
    timestamp_us: int | None = None
    exchange_id: object | None = None

    @property
    def utf8_size(self) -> int:
        time_label = _format_minute(self.timestamp_us) if self.timestamp_us is not None else "unknown"
        return len(self.user.encode("utf-8")) + len(self.assistant.encode("utf-8")) + len(time_label.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _BackgroundSection:
    text: str
    through_sequence: int
    earliest_us: int | None = None
    latest_us: int | None = None
    has_unknown_time: bool = False

    @property
    def recency(self) -> str:
        return _recency(self.earliest_us, self.latest_us, self.has_unknown_time)


@dataclass(frozen=True, slots=True)
class _ContextRecord:
    sequence: int
    text: str
    timestamp_us: int | None

    @property
    def utf8_size(self) -> int:
        time_label = _format_minute(self.timestamp_us) if self.timestamp_us is not None else "unknown"
        return len(self.text.encode("utf-8")) + len(time_label.encode("utf-8"))


@dataclass(slots=True)
class _ParticipantMemory:
    clearing: bool = False
    cleanup: asyncio.Task[None] | None = None
    transcript: list[_TranscriptEntry] = field(default_factory=list)
    context_records: dict[str, list[_ContextRecord]] = field(default_factory=dict)
    backgrounds: dict[str, _BackgroundSection] = field(default_factory=dict)
    dialogue_background: _BackgroundSection = field(default_factory=lambda: _BackgroundSection("", -1))
    revisions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dialogue_revision: int = 0
    # Only exchanges not yet compacted stay in ``transcript``; ``seen`` keeps
    # identities of recent exchanges so a front-end window that still contains
    # compacted exchanges is not replayed.
    next_dialogue_sequence: int = 0
    next_record_sequence: int = 0
    seen: deque[_Exchange] = field(default_factory=lambda: deque(maxlen=_SEEN_EXCHANGES))
    dialogue_job: asyncio.Task[bool] | None = None
    context_jobs: dict[str, asyncio.Task[bool]] = field(default_factory=dict)
    context_order: list[str] = field(default_factory=list)
    retired_jobs: set[asyncio.Task[bool]] = field(default_factory=set)


# Compactors receive the rendered earlier summary, new entries (empty when only
# condensing an oversized summary) and the byte limit for the result.
_Compactor = Callable[[str, str, Sequence[_TranscriptEntry], int], Awaitable[str]]
_DialogueCompactor = Callable[[str, Sequence[_TranscriptEntry], int], Awaitable[str]]
_logger = logging.getLogger(__name__)
_Exchange = tuple[str, str, int | None, object | None]
_SEEN_EXCHANGES = 256
# After a failed compaction, uncompacted dialogue or source updates beyond this
# multiple of the trigger size are dropped oldest first.
_STORAGE_FACTOR = 8


def _exchange_key(user: str, assistant: str, timestamp_us: int | None, exchange_id: object | None) -> _Exchange:
    return (_digest(user), _digest(assistant), timestamp_us, exchange_id)


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _same_exchange(left: _Exchange, right: _Exchange) -> bool:
    """Compare exchange identity, falling back only when both sides lack it."""

    if left[3] is not None or right[3] is not None:
        return left[3] is not None and left[3] is right[3]
    if left[2] is not None and right[2] is not None:
        return left[2] == right[2] and left[:2] == right[:2]
    return left[:2] == right[:2]


def _contains_exchange_window(existing: tuple[_Exchange, ...], incoming: tuple[_Exchange, ...]) -> bool:
    """Return whether a bounded input window is already retained."""

    return len(incoming) <= len(existing) and any(
        all(_same_exchange(left, right) for left, right in zip(existing[start : start + len(incoming)], incoming))
        for start in range(len(existing) - len(incoming) + 1)
    )


class _ConversationContext:
    """Retain recent turns and compact old entries independently by source.

    Dialogue and each incremental source use a rolling summary: entries
    accumulate until they pass the trigger size, then the model merges the
    previous summary with the oldest entries into a new summary and those
    entries are discarded.

    ``payload_budget_bytes`` bounds the prior background and transcript text
    supplied to one compaction request. It does not include instructions or
    transport framing and is not a model-token estimate. Callers provide it
    from their own model/configuration boundary.
    """

    def __init__(self) -> None:
        self._participants: dict[str, _ParticipantMemory] = {}
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    def activate(self, participant: str) -> None:
        memory = self._participants.get(participant)
        if self._closing or (memory is not None and memory.clearing):
            raise RuntimeError("conversation context cleanup is still in progress")

    def _memory(self, participant: str) -> _ParticipantMemory:
        self.activate(participant)
        return self._participants.setdefault(participant, _ParticipantMemory())

    def append(
        self,
        participant: str,
        user: str,
        assistant: str,
        *,
        timestamp_us: int | None = None,
        exchange_id: object | None = None,
    ) -> _TranscriptEntry:
        memory = self._memory(participant)
        entry = _TranscriptEntry(memory.next_dialogue_sequence, "dialogue", user, assistant, timestamp_us, exchange_id)
        memory.next_dialogue_sequence += 1
        memory.transcript.append(entry)
        memory.seen.append(_exchange_key(user, assistant, timestamp_us, exchange_id))
        return entry

    def reconcile_transcript(
        self,
        participant: str,
        exchanges: Sequence[tuple[str, str] | tuple[str, str, int | None]],
        *,
        exchange_ids: Sequence[object | None] = (),
    ) -> None:
        """Merge a front-end history window without replaying retained exchanges."""

        memory = self._memory(participant)
        existing = tuple(memory.seen)
        if exchange_ids and len(exchange_ids) != len(exchanges):
            raise ValueError("history exchange IDs must align with the exchanges")
        normalized = tuple(
            (
                exchange[0],
                exchange[1],
                exchange[2] if len(exchange) == 3 else None,
                exchange_ids[index] if exchange_ids else None,
            )
            for index, exchange in enumerate(exchanges)
        )
        if not normalized:
            return
        keys = tuple(_exchange_key(*exchange) for exchange in normalized)

        # FE history is a bounded recent window and can start after every turn
        # retained here. Match its overlap with our tail; a non-overlapping
        # window is new completed context, not a divergent transcript to drop.
        if _contains_exchange_window(existing, keys):
            return
        overlap = min(len(existing), len(keys))
        while overlap and not all(
            _same_exchange(left, right) for left, right in zip(existing[-overlap:], keys[:overlap])
        ):
            overlap -= 1
        for user, assistant, timestamp_us, exchange_id in normalized[overlap:]:
            self.append(participant, user, assistant, timestamp_us=timestamp_us, exchange_id=exchange_id)

    def transcript(self, participant: str) -> tuple[_TranscriptEntry, ...]:
        """Return the exchanges not yet compacted into the dialogue summary."""

        memory = self._participants.get(participant)
        return () if memory is None else tuple(memory.transcript)

    def dialogue_snapshot(self, participant: str) -> tuple[str, tuple[_TranscriptEntry, ...]]:
        memory = self._participants.get(participant)
        if memory is None:
            return "", ()
        summary = memory.dialogue_background
        recent = tuple(entry for entry in memory.transcript if entry.sequence > summary.through_sequence)
        return summary.text, recent

    def dialogue_recency(self, participant: str) -> str:
        memory = self._participants.get(participant)
        if memory is None:
            return "recorded time unknown"
        return memory.dialogue_background.recency

    def schedule_dialogue_compaction(
        self,
        participant: str,
        *,
        trigger_budget_bytes: int,
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _DialogueCompactor,
    ) -> asyncio.Task[bool] | None:
        """Compact the oldest exchanges into the summary when unsummarized dialogue is large.

        ``summary_budget_bytes`` must leave room in ``payload_budget_bytes``
        for new exchanges; one exchange larger than the remaining room is
        clipped for the request rather than blocking compaction.
        """

        memory = self._participants.get(participant)
        if memory is None or memory.clearing:
            return None
        current = memory.dialogue_job
        if current is not None and not current.done():
            return current
        previous = memory.dialogue_background
        pending = tuple(memory.transcript)
        pending_bytes = sum(entry.utf8_size for entry in pending)
        if pending_bytes <= trigger_budget_bytes:
            return None
        # The newest complete exchange remains verbatim for the next turn.
        # Compact only the immutable prefix before it.
        prefix = pending[:-1]
        if not prefix:
            return None
        previous_input = _summary_for_compaction(previous, time_label="exchange completion times")
        available_bytes = max(payload_budget_bytes - len(previous_input.encode("utf-8")), 1)
        batch = _clip_single_entry(_oldest_batch(prefix, available_bytes), available_bytes)
        revision = memory.dialogue_revision
        task = asyncio.create_task(
            self._compact_dialogue(
                participant,
                revision,
                previous,
                previous_input,
                batch,
                payload_budget_bytes,
                summary_budget_bytes,
                compact,
            ),
            name=f"conversation-dialogue-compact:{participant}",
            context=nemo_relay.fork_asyncio_context(),
        )
        memory.dialogue_job = task
        task.add_done_callback(
            lambda done: self._forget_dialogue_job(
                participant,
                done,
                trigger_budget_bytes=trigger_budget_bytes,
                payload_budget_bytes=payload_budget_bytes,
                summary_budget_bytes=summary_budget_bytes,
                compact=compact,
            )
        )
        return task

    async def _compact_dialogue(
        self,
        participant: str,
        revision: int,
        previous: _BackgroundSection,
        previous_input: str,
        batch: tuple[_TranscriptEntry, ...],
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _DialogueCompactor,
    ) -> bool:
        summary = await _bounded_summary(
            participant,
            "dialogue",
            await compact(previous_input, batch, summary_budget_bytes),
            lambda text: compact(text, (), summary_budget_bytes),
            summary_budget_bytes,
            payload_budget_bytes,
        )
        memory = self._participants.get(participant)
        # Commit only while the prior summary is unchanged. Exchanges are only
        # appended, so the summary covers everything through the batch's last
        # sequence and newer exchanges stay verbatim.
        if memory is None or memory.clearing or memory.dialogue_revision != revision:
            return False
        if memory.dialogue_background != previous:
            return False
        through = batch[-1].sequence
        memory.dialogue_background = _merge_background(summary, through, previous, batch)
        memory.transcript = [entry for entry in memory.transcript if entry.sequence > through]
        return True

    def source_snapshots(
        self, participant: str
    ) -> tuple[tuple[str, _BackgroundSection | None, tuple[_ContextRecord, ...]], ...]:
        """Return each source's summary and its uncompacted updates, in source order."""

        memory = self._participants.get(participant)
        if memory is None:
            return ()
        order = list(memory.context_order)
        order.extend(source for source in memory.backgrounds if source not in order)
        snapshots = []
        for source in order:
            background = memory.backgrounds.get(source)
            records = tuple(memory.context_records.get(source, ()))
            if background is not None or records:
                snapshots.append((source, background, records))
        return tuple(snapshots)

    def sections(self, participant: str) -> tuple[tuple[str, str], ...]:
        return tuple((source, text) for source, text, _ in self.sections_with_recency(participant))

    def sections_with_recency(self, participant: str) -> tuple[tuple[str, str, str], ...]:
        memory = self._participants.get(participant)
        if memory is None:
            return ()
        sections = dict((source, section.text) for source, section in memory.backgrounds.items())
        for source, records in memory.context_records.items():
            if records:
                sections[source] = "\n".join(
                    filter(None, (sections.get(source, ""), *(record.text for record in records)))
                )
        order = list(memory.context_order)
        order.extend(source for source in memory.backgrounds if source not in order)
        rendered = []
        for source in order:
            if source not in sections:
                continue
            background = memory.backgrounds.get(source, _BackgroundSection("", -1))
            records = memory.context_records.get(source, ())
            times = [record.timestamp_us for record in records]
            known = [timestamp for timestamp in times if timestamp is not None]
            if background.earliest_us is not None:
                known.append(background.earliest_us)
            if background.latest_us is not None:
                known.append(background.latest_us)
            recency = _recency(
                min(known) if known else None,
                max(known) if known else None,
                background.has_unknown_time or any(timestamp is None for timestamp in times),
            )
            rendered.append((source, sections[source], recency))
        return tuple(rendered)

    async def update(
        self,
        participant: str,
        source: str,
        mode: str,
        text: str,
        *,
        context_budget_bytes: int,
        timestamp_us: int | None = None,
    ) -> tuple[bool, bool]:
        """Store an authoritative summary or append a source update.

        A summary whose rendered size exceeds ``context_budget_bytes`` clears
        the source. Incremental records are always kept until compaction folds
        them into the source summary.
        """

        memory = self._participants.get(participant)
        if self._closing or (memory is not None and memory.clearing):
            return False, False
        memory = self._memory(participant)
        if mode not in {"summary", "incremental"}:
            raise ValueError("context mode must be 'incremental' or 'summary'")
        if mode == "summary" and _rendered_source_size(source, text, timestamp_us) > context_budget_bytes:
            self._cancel_source_jobs(memory, source)
            memory.revisions[source] += 1
            memory.backgrounds.pop(source, None)
            memory.context_records.pop(source, None)
            return False, True
        if mode == "summary":
            self._cancel_source_jobs(memory, source)
            memory.revisions[source] += 1
            if source not in memory.context_order:
                memory.context_order.append(source)
            memory.backgrounds[source] = _BackgroundSection(text, -1, timestamp_us, timestamp_us, timestamp_us is None)
            memory.context_records.pop(source, None)
            return True, False
        if source not in memory.context_order:
            memory.context_order.append(source)
        memory.context_records.setdefault(source, []).append(
            _ContextRecord(memory.next_record_sequence, text, timestamp_us)
        )
        memory.next_record_sequence += 1
        return True, False

    async def drop_source(self, participant: str, source: str) -> None:
        memory = self._participants.get(participant)
        if memory is None:
            return
        self._cancel_source_jobs(memory, source)
        memory.revisions[source] += 1
        memory.backgrounds.pop(source, None)
        memory.context_records.pop(source, None)
        if source in memory.context_order:
            memory.context_order.remove(source)

    def schedule_context_compaction(
        self,
        participant: str,
        source: str,
        *,
        trigger_budget_bytes: int,
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _Compactor,
    ) -> asyncio.Task[bool] | None:
        """Compact a source's oldest updates into its summary once they exceed the trigger size.

        Runs without blocking delivery. As for dialogue, the summary limit must
        leave room in the request, and one update larger than the remaining
        room is clipped for the request.
        """

        memory = self._participants.get(participant)
        if memory is None or memory.clearing or not memory.context_records.get(source):
            return None
        current = memory.context_jobs.get(source)
        if current is not None and not current.done():
            return current
        records = tuple(memory.context_records[source])
        if sum(record.utf8_size for record in records) <= trigger_budget_bytes:
            return None
        previous = memory.backgrounds.get(source, _BackgroundSection("", -1))
        previous_input = _summary_for_compaction(previous, time_label="publication times")
        available_bytes = max(payload_budget_bytes - len(previous_input.encode("utf-8")), 1)
        entries = tuple(
            _TranscriptEntry(record.sequence, source, record.text, "", record.timestamp_us) for record in records
        )
        batch = _clip_single_entry(_oldest_batch(entries, available_bytes), available_bytes)
        revision = memory.revisions[source]
        task = asyncio.create_task(
            self._compact_context(
                participant,
                source,
                revision,
                previous,
                previous_input,
                batch,
                payload_budget_bytes,
                summary_budget_bytes,
                compact,
            ),
            name=f"conversation-context-compact:{participant}:{source}",
            context=nemo_relay.fork_asyncio_context(),
        )
        memory.context_jobs[source] = task
        task.add_done_callback(
            lambda done: self._forget_context_job(
                participant,
                source,
                done,
                trigger_budget_bytes=trigger_budget_bytes,
                payload_budget_bytes=payload_budget_bytes,
                summary_budget_bytes=summary_budget_bytes,
                compact=compact,
            )
        )
        return task

    async def _compact_context(
        self,
        participant: str,
        source: str,
        revision: int,
        previous: _BackgroundSection,
        previous_input: str,
        batch: tuple[_TranscriptEntry, ...],
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _Compactor,
    ) -> bool:
        summary = await _bounded_summary(
            participant,
            source,
            await compact(source, previous_input, batch, summary_budget_bytes),
            lambda text: compact(source, text, (), summary_budget_bytes),
            summary_budget_bytes,
            payload_budget_bytes,
        )
        memory = self._participants.get(participant)
        # A summary replacement or source drop changes the revision; records
        # are otherwise only appended, so the summary covers everything
        # through the batch's last sequence and later records stay verbatim.
        if memory is None or memory.clearing or memory.revisions[source] != revision:
            return False
        if memory.backgrounds.get(source, _BackgroundSection("", -1)) != previous:
            return False
        through = batch[-1].sequence
        memory.backgrounds[source] = _merge_background(summary, through, previous, batch)
        tail = [record for record in memory.context_records.get(source, ()) if record.sequence > through]
        if tail:
            memory.context_records[source] = tail
        else:
            memory.context_records.pop(source, None)
        return True

    async def clear(self, participant: str) -> None:
        memory = self._participants.get(participant)
        if memory is None:
            return
        if memory.cleanup is None:
            memory.clearing = True
            memory.cleanup = asyncio.create_task(
                self._clear(participant, memory),
                name=f"conversation-clear:{participant}",
                context=nemo_relay.fork_asyncio_context(),
            )
        await _join_cleanup(memory.cleanup)

    async def _clear(self, participant: str, memory: _ParticipantMemory) -> None:
        tasks = (
            (() if memory.dialogue_job is None else (memory.dialogue_job,))
            + tuple(memory.context_jobs.values())
            + tuple(memory.retired_jobs)
        )
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        if tasks:
            # Background-task supervisors report failures; cleanup owns their termination.
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._participants.get(participant) is memory:
            del self._participants[participant]

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close(), name="conversation-context-close", context=nemo_relay.fork_asyncio_context()
            )
        await _join_cleanup(self._close_task)

    async def _close(self) -> None:
        await asyncio.gather(*(self.clear(participant) for participant in tuple(self._participants)))

    def _forget_dialogue_job(
        self,
        participant: str,
        task: asyncio.Task[bool],
        *,
        trigger_budget_bytes: int,
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _DialogueCompactor,
    ) -> None:
        _log_task_error(participant, "dialogue", task)
        memory = self._participants.get(participant)
        # Act only on the memory that still owns this job, never on a later
        # session's memory for the same participant.
        if memory is None or memory.dialogue_job is not task:
            return
        memory.dialogue_job = None
        if not task.cancelled() and task.exception() is not None:
            # Failed exchanges stay for the next attempt, bounded so a model
            # that keeps failing cannot grow memory without limit.
            _drop_oldest_beyond(participant, "dialogue", memory.transcript, trigger_budget_bytes * _STORAGE_FACTOR)
        if not memory.clearing and not task.cancelled() and task.exception() is None and task.result():
            self.schedule_dialogue_compaction(
                participant,
                trigger_budget_bytes=trigger_budget_bytes,
                payload_budget_bytes=payload_budget_bytes,
                summary_budget_bytes=summary_budget_bytes,
                compact=compact,
            )

    def _forget_context_job(
        self,
        participant: str,
        source: str,
        task: asyncio.Task[bool],
        *,
        trigger_budget_bytes: int,
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _Compactor,
    ) -> None:
        _log_task_error(participant, source, task)
        memory = self._participants.get(participant)
        # As for dialogue, a retired or superseded job does not touch the source.
        if memory is None or memory.context_jobs.get(source) is not task:
            return
        memory.context_jobs.pop(source, None)
        if not task.cancelled() and task.exception() is not None:
            records = memory.context_records.get(source, [])
            _drop_oldest_beyond(participant, source, records, trigger_budget_bytes * _STORAGE_FACTOR)
        if (
            not memory.clearing
            and not task.cancelled()
            and task.exception() is None
            and memory.context_records.get(source)
        ):
            self.schedule_context_compaction(
                participant,
                source,
                trigger_budget_bytes=trigger_budget_bytes,
                payload_budget_bytes=payload_budget_bytes,
                summary_budget_bytes=summary_budget_bytes,
                compact=compact,
            )

    @staticmethod
    def _cancel_source_jobs(memory: _ParticipantMemory, source: str) -> None:
        task = memory.context_jobs.pop(source, None)
        if task is not None and not task.done():
            memory.retired_jobs.add(task)
            task.add_done_callback(memory.retired_jobs.discard)
            task.cancel()


async def _join_cleanup(task: asyncio.Task[None]) -> None:
    """Keep the cleanup owner alive and join it before propagating waiter cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is None or not current.cancelling():
                raise
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


def _log_task_error(participant: str, source: str, task: asyncio.Task[bool]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        _logger.error(
            "conversation context compaction failed participant=%r source=%r",
            participant,
            source,
            exc_info=(type(error), error, error.__traceback__),
        )


def _oldest_batch(entries: tuple[_TranscriptEntry, ...], input_budget_bytes: int) -> tuple[_TranscriptEntry, ...]:
    """Take the oldest entries that fit; the first entry is always taken so compaction progresses."""

    batch = []
    size = 0
    for entry in entries:
        if batch and size + entry.utf8_size > input_budget_bytes:
            break
        batch.append(entry)
        size += entry.utf8_size
        if size >= input_budget_bytes:
            break
    return tuple(batch)


def _clip_single_entry(batch: tuple[_TranscriptEntry, ...], limit_bytes: int) -> tuple[_TranscriptEntry, ...]:
    """Clip a lone exchange that exceeds the request budget, keeping its start and end."""

    if len(batch) != 1 or batch[0].utf8_size <= limit_bytes:
        return batch
    entry = batch[0]
    half = limit_bytes // 2
    return (replace(entry, user=_clip(entry.user, half), assistant=_clip(entry.assistant, half)),)


def _clip(text: str, limit_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit_bytes:
        return text
    marker = f"\n[... {len(data) - limit_bytes} bytes omitted ...]\n".encode()
    room = limit_bytes - len(marker)
    if room <= 0:
        return _truncate(text, limit_bytes)
    head = room // 2
    return (data[:head] + marker + data[len(data) - (room - head) :]).decode("utf-8", "ignore")


def _truncate(text: str, limit_bytes: int) -> str:
    return text.encode("utf-8")[: max(limit_bytes, 0)].decode("utf-8", "ignore")


def _drop_oldest_beyond(
    participant: str,
    source: str,
    entries: list[_TranscriptEntry] | list[_ContextRecord],
    limit_bytes: int,
) -> None:
    """Bound uncompacted entries while compaction keeps failing; the newest entry is kept."""

    while len(entries) > 1 and sum(entry.utf8_size for entry in entries) > limit_bytes:
        dropped = entries.pop(0)
        _logger.error(
            "dropping uncompacted context entry participant=%r source=%r size_bytes=%d limit_bytes=%d",
            participant,
            source,
            dropped.utf8_size,
            limit_bytes,
        )


async def _bounded_summary(
    participant: str,
    source: str,
    summary: str,
    condense: Callable[[str], Awaitable[str]],
    limit_bytes: int,
    payload_budget_bytes: int,
) -> str:
    """Return a non-empty summary within ``limit_bytes``.

    An oversized summary is condensed once by the model and truncated if it
    still overshoots, so a summary can never block later compaction.
    """

    summary = summary.strip()
    if len(summary.encode("utf-8")) > limit_bytes:
        _logger.warning(
            "condensing oversized context summary participant=%r source=%r size_bytes=%d budget_bytes=%d",
            participant,
            source,
            len(summary.encode("utf-8")),
            limit_bytes,
        )
        summary = _truncate((await condense(_clip(summary, payload_budget_bytes))).strip(), limit_bytes)
    if not summary:
        raise ValueError("context compaction returned an empty summary")
    return summary


def _rendered_source_size(source: str, text: str, timestamp_us: int | None = None) -> int:
    recency = _recency(timestamp_us, timestamp_us, timestamp_us is None)
    rendered = f"Source {source} (publication time {recency}):\n{text}"
    return len(rendered.encode("utf-8"))


def _merge_background(
    text: str,
    through_sequence: int,
    previous: _BackgroundSection,
    entries: Sequence[_TranscriptEntry] | Sequence[_ContextRecord],
) -> _BackgroundSection:
    times = [entry.timestamp_us for entry in entries]
    known = [timestamp for timestamp in times if timestamp is not None]
    if previous.earliest_us is not None:
        known.append(previous.earliest_us)
    if previous.latest_us is not None:
        known.append(previous.latest_us)
    return _BackgroundSection(
        text,
        through_sequence,
        min(known) if known else None,
        max(known) if known else None,
        previous.has_unknown_time or any(timestamp is None for timestamp in times) or not known,
    )


def _recency(earliest_us: int | None, latest_us: int | None, has_unknown: bool) -> str:
    if earliest_us is None or latest_us is None:
        return "unknown"
    first, last = _format_minute(earliest_us), _format_minute(latest_us)
    label = f"around {first}" if first == last else f"from around {first} through around {last}"
    if has_unknown:
        label += "; some included entries have unknown times"
    return label


def _summary_for_compaction(section: _BackgroundSection, *, time_label: str) -> str:
    if not section.text:
        return ""
    return f"Earlier summary ({time_label} {section.recency}):\n{section.text}"
