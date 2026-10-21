# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private participant dialogue and source-context storage.

Only modules in ``xr_ai_conversation`` use this implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TypeVar

import nemo_relay


def _format_minute(timestamp_us: int) -> str:
    """Format an exchange timestamp at the conversation's display precision."""

    return datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).strftime("%Y-%m-%d %H:%M UTC")


def _time_label(timestamp_us: int | None) -> str:
    return f"around {_format_minute(timestamp_us)}" if timestamp_us is not None else "at an unknown time"


@dataclass(frozen=True, slots=True)
class _TranscriptEntry:
    sequence: int
    user: str
    assistant: str
    timestamp_us: int | None = None
    exchange_id: object | None = None

    @property
    def utf8_size(self) -> int:
        """Return the size of this exchange as rendered into a prompt."""

        return len(_exchange_text(self).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _ContextRecord:
    sequence: int
    source: str
    text: str
    timestamp_us: int | None

    @property
    def utf8_size(self) -> int:
        """Return the size of this update as rendered into a prompt."""

        return len(_update_text(self).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _BackgroundSection:
    """A summary, or a summary topic's current state, and the time of its newest content."""

    text: str
    through_sequence: int
    latest_us: int | None = None


@dataclass(slots=True)
class _ParticipantMemory:
    # Only exchanges not yet compacted stay in ``transcript``; ``seen`` keeps
    # the identity of every exchange so a front-end window that still contains
    # compacted exchanges is not replayed.
    transcript: list[_TranscriptEntry] = field(default_factory=list)
    seen: list[_Exchange] = field(default_factory=list)
    next_dialogue_sequence: int = 0
    dialogue_background: _BackgroundSection = field(default_factory=lambda: _BackgroundSection("", -1))
    dialogue_job: asyncio.Task[bool] | None = None
    context_records: dict[str, list[_ContextRecord]] = field(default_factory=dict)
    next_record_sequence: int = 0
    backgrounds: dict[str, _BackgroundSection] = field(default_factory=dict)
    context_jobs: dict[str, asyncio.Task[bool]] = field(default_factory=dict)
    context_order: list[str] = field(default_factory=list)


# Compactors receive the rendered earlier summary, new entries (empty when only
# condensing an oversized summary) and the byte limit for the result.
_Compactor = Callable[[str, str, Sequence[_ContextRecord], int], Awaitable[str]]
_DialogueCompactor = Callable[[str, Sequence[_TranscriptEntry], int], Awaitable[str]]
_logger = logging.getLogger(__name__)


class _TruncatedSummary(RuntimeError):
    """A compaction response cut off at the model's output limit."""


_Exchange = tuple[str, str, int | None, object | None]
_Entry = TypeVar("_Entry", _TranscriptEntry, _ContextRecord)


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


def _contains_exchange_window(existing: Sequence[_Exchange], incoming: Sequence[_Exchange]) -> bool:
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

    ``payload_budget_bytes`` bounds the prior summary and entries supplied to
    one compaction request. It does not include instructions or transport
    framing and is not a model-token estimate.
    """

    def __init__(self) -> None:
        self._participants: dict[str, _ParticipantMemory] = {}

    def _memory(self, participant: str) -> _ParticipantMemory:
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
        entry = _TranscriptEntry(memory.next_dialogue_sequence, user, assistant, timestamp_us, exchange_id)
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
        """Merge a front-end history window without replaying known exchanges."""

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
        memory = self._memory(participant)
        keys = tuple(_exchange_key(*exchange) for exchange in normalized)
        # FE history is a bounded recent window, so only the recent end of the
        # session needs comparing. It can start after every exchange known
        # here; a non-overlapping window is new completed context.
        existing = memory.seen[-2 * len(keys) :]
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

    def dialogue_snapshot(self, participant: str) -> tuple[_BackgroundSection, tuple[_TranscriptEntry, ...]]:
        """Return the dialogue summary and the exchanges not yet compacted into it."""

        memory = self._participants.get(participant)
        if memory is None:
            return _BackgroundSection("", -1), ()
        return memory.dialogue_background, tuple(memory.transcript)

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
        if memory is None:
            return None
        current = memory.dialogue_job
        if current is not None and not current.done():
            return current
        pending = tuple(memory.transcript)
        if sum(entry.utf8_size for entry in pending) <= trigger_budget_bytes:
            return None
        # The newest complete exchange remains verbatim for the next turn.
        # Compact only the immutable prefix before it.
        prefix = pending[:-1]
        if not prefix:
            return None
        previous_input = _summary_for_compaction(memory.dialogue_background)
        available_bytes = max(payload_budget_bytes - len(previous_input.encode("utf-8")), 1)
        batch = _oldest_batch(prefix, available_bytes)
        task = asyncio.create_task(
            self._compact_dialogue(
                participant,
                memory,
                previous_input,
                batch,
                _clip_single_entry(batch, available_bytes),
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
        memory: _ParticipantMemory,
        previous_input: str,
        batch: tuple[_TranscriptEntry, ...],
        request_batch: tuple[_TranscriptEntry, ...],
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _DialogueCompactor,
    ) -> bool:
        summary = await _bounded_summary(
            participant,
            "dialogue",
            await compact(previous_input, request_batch, summary_budget_bytes),
            lambda text: compact(text, (), summary_budget_bytes),
            summary_budget_bytes,
            payload_budget_bytes,
        )
        if self._participants.get(participant) is not memory:
            return False
        # Exchanges are only appended, so the summary covers everything through
        # the batch's last sequence and newer exchanges stay verbatim.
        through = batch[-1].sequence
        memory.dialogue_background = _BackgroundSection(summary, through, _latest(memory.dialogue_background, batch))
        memory.transcript = [entry for entry in memory.transcript if entry.sequence > through]
        return True

    def source_snapshots(
        self, participant: str
    ) -> tuple[tuple[str, _BackgroundSection | None, tuple[_ContextRecord, ...]], ...]:
        """Return each source's summary or state and its uncompacted updates, in source order."""

        memory = self._participants.get(participant)
        if memory is None:
            return ()
        snapshots = []
        for source in memory.context_order:
            background = memory.backgrounds.get(source)
            records = tuple(memory.context_records.get(source, ()))
            if background is not None or records:
                snapshots.append((source, background, records))
        return tuple(snapshots)

    def sections(self, participant: str) -> tuple[tuple[str, str], ...]:
        """Return each source's stored text: its summary or state followed by uncompacted updates."""

        return tuple(
            (
                source,
                "\n".join(filter(None, ("" if background is None else background.text, *(r.text for r in records)))),
            )
            for source, background, records in self.source_snapshots(participant)
        )

    async def update(
        self,
        participant: str,
        source: str,
        mode: str,
        text: str,
        *,
        timestamp_us: int | None = None,
    ) -> None:
        """Replace a summary source's state or append an incremental source update."""

        if mode not in {"summary", "incremental"}:
            raise ValueError("context mode must be 'incremental' or 'summary'")
        memory = self._memory(participant)
        if source not in memory.context_order:
            memory.context_order.append(source)
        if mode == "summary":
            memory.backgrounds[source] = _BackgroundSection(text, -1, timestamp_us)
            return
        memory.context_records.setdefault(source, []).append(
            _ContextRecord(memory.next_record_sequence, source, text, timestamp_us)
        )
        memory.next_record_sequence += 1

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
        if memory is None or not memory.context_records.get(source):
            return None
        current = memory.context_jobs.get(source)
        if current is not None and not current.done():
            return current
        records = tuple(memory.context_records[source])
        if sum(record.utf8_size for record in records) <= trigger_budget_bytes:
            return None
        previous = memory.backgrounds.get(source, _BackgroundSection("", -1))
        previous_input = _summary_for_compaction(previous)
        available_bytes = max(payload_budget_bytes - len(previous_input.encode("utf-8")), 1)
        batch = _oldest_batch(records, available_bytes)
        task = asyncio.create_task(
            self._compact_context(
                participant,
                memory,
                source,
                previous_input,
                batch,
                _clip_single_entry(batch, available_bytes),
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
        memory: _ParticipantMemory,
        source: str,
        previous_input: str,
        batch: tuple[_ContextRecord, ...],
        request_batch: tuple[_ContextRecord, ...],
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _Compactor,
    ) -> bool:
        summary = await _bounded_summary(
            participant,
            source,
            await compact(source, previous_input, request_batch, summary_budget_bytes),
            lambda text: compact(source, text, (), summary_budget_bytes),
            summary_budget_bytes,
            payload_budget_bytes,
        )
        if self._participants.get(participant) is not memory:
            return False
        # Updates are only appended, so the summary covers everything through
        # the batch's last sequence and later updates stay verbatim.
        through = batch[-1].sequence
        previous = memory.backgrounds.get(source, _BackgroundSection("", -1))
        memory.backgrounds[source] = _BackgroundSection(summary, through, _latest(previous, batch))
        tail = [record for record in memory.context_records.get(source, ()) if record.sequence > through]
        if tail:
            memory.context_records[source] = tail
        else:
            memory.context_records.pop(source, None)
        return True

    async def clear(self, participant: str) -> None:
        """Forget a participant, cancelling and awaiting its compaction jobs."""

        memory = self._participants.pop(participant, None)
        if memory is None:
            return
        jobs = [
            job for job in (memory.dialogue_job, *memory.context_jobs.values()) if job is not None and not job.done()
        ]
        for job in jobs:
            job.cancel()
        # Completion callbacks report failures; cleanup only waits for termination.
        await asyncio.gather(*jobs, return_exceptions=True)

    async def close(self) -> None:
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
        # Act only on the memory that still owns this job.
        if memory is None or memory.dialogue_job is not task:
            return
        memory.dialogue_job = None
        # Continue while the section is still over its trigger. A failure
        # leaves the exchanges for the attempt after the next exchange.
        if not task.cancelled() and task.exception() is None and task.result():
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
        if memory is None or memory.context_jobs.get(source) is not task:
            return
        memory.context_jobs.pop(source)
        if not task.cancelled() and task.exception() is None and task.result():
            self.schedule_context_compaction(
                participant,
                source,
                trigger_budget_bytes=trigger_budget_bytes,
                payload_budget_bytes=payload_budget_bytes,
                summary_budget_bytes=summary_budget_bytes,
                compact=compact,
            )


def _log_task_error(participant: str, source: str, task: asyncio.Task[bool]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        _logger.error(
            "conversation context compaction failed participant=%r source=%r",
            participant,
            source,
            exc_info=(type(error), error, error.__traceback__),
        )


def _oldest_batch(entries: Sequence[_Entry], input_budget_bytes: int) -> tuple[_Entry, ...]:
    """Take the oldest entries that fit; the first entry is always taken so compaction progresses."""

    batch: list[_Entry] = []
    size = 0
    for entry in entries:
        if batch and size + entry.utf8_size > input_budget_bytes:
            break
        batch.append(entry)
        size += entry.utf8_size
        if size >= input_budget_bytes:
            break
    return tuple(batch)


def _clip_single_entry(batch: tuple[_Entry, ...], limit_bytes: int) -> tuple[_Entry, ...]:
    """Clip a lone entry that exceeds the request budget, keeping its start and end."""

    if len(batch) != 1 or batch[0].utf8_size <= limit_bytes:
        return batch
    entry = batch[0]
    # The limit covers the entry's framing as well as its text.
    if isinstance(entry, _ContextRecord):
        room = max(limit_bytes - (entry.utf8_size - len(entry.text.encode("utf-8"))), 0)
        return (replace(entry, text=_clip(entry.text, room)),)
    text_bytes = len(entry.user.encode("utf-8")) + len(entry.assistant.encode("utf-8"))
    half = max(limit_bytes - (entry.utf8_size - text_bytes), 0) // 2
    return (replace(entry, user=_clip(entry.user, half), assistant=_clip(entry.assistant, half)),)


def _clip(text: str, limit_bytes: int) -> str:
    """Keep the start and end of ``text`` within ``limit_bytes``, marking the omission."""

    data = text.encode("utf-8")
    if len(data) <= limit_bytes:
        return text
    marker = f"\n[... {len(data) - limit_bytes} bytes omitted ...]\n".encode()
    room = limit_bytes - len(marker)
    if room <= 0:
        return data[: max(limit_bytes, 0)].decode("utf-8", "ignore")
    head = room // 2
    return (data[:head] + marker + data[len(data) - (room - head) :]).decode("utf-8", "ignore")


async def _bounded_summary(
    participant: str,
    source: str,
    summary: str,
    condense: Callable[[str], Awaitable[str]],
    limit_bytes: int,
    payload_budget_bytes: int,
) -> str:
    """Return a non-empty summary within ``limit_bytes``.

    An oversized summary is condensed once by the model. If the condensed
    summary is cut off or still overshoots, the summary keeps its start and
    end, so a summary can never block later compaction.
    """

    summary = summary.strip()
    if not summary:
        raise ValueError("context compaction returned an empty summary")
    if len(summary.encode("utf-8")) <= limit_bytes:
        return summary
    _logger.warning(
        "condensing oversized context summary participant=%r source=%r size_bytes=%d budget_bytes=%d",
        participant,
        source,
        len(summary.encode("utf-8")),
        limit_bytes,
    )
    try:
        condensed = (await condense(_clip(summary, payload_budget_bytes))).strip()
    except _TruncatedSummary:
        _logger.warning("condensed context summary was cut off participant=%r source=%r", participant, source)
        condensed = ""
    return _clip(condensed or summary, limit_bytes)


def _latest(previous: _BackgroundSection, entries: Sequence[_TranscriptEntry] | Sequence[_ContextRecord]) -> int | None:
    times = [entry.timestamp_us for entry in entries if entry.timestamp_us is not None]
    if previous.latest_us is not None:
        times.append(previous.latest_us)
    return max(times, default=None)


def _summary_for_compaction(section: _BackgroundSection) -> str:
    if not section.text:
        return ""
    return f"Earlier summary (up to {_time_label(section.latest_us)}):\n{section.text}"


# Prompt rendering lives beside storage so compaction triggers measure entries
# exactly as they appear in a prompt.


def _exchange_text(entry: _TranscriptEntry, field_limit_bytes: int | None = None) -> str:
    user, assistant = entry.user, entry.assistant
    if field_limit_bytes is not None:
        user, assistant = _clip(user, field_limit_bytes), _clip(assistant, field_limit_bytes)
    return f"Exchange completed {_time_label(entry.timestamp_us)}:\nUser: {user}\nAssistant: “{assistant}”"


def _update_text(record: _ContextRecord, limit_bytes: int | None = None) -> str:
    text = record.text if limit_bytes is None else _clip(record.text, limit_bytes)
    return f"Source {record.source} update, published {_time_label(record.timestamp_us)}:\n{text}"


def _state_text(source: str, text: str, timestamp_us: int | None) -> str:
    return f"Source {source} current state, published {_time_label(timestamp_us)}:\n{text}"


def _dialogue_summary_text(section: _BackgroundSection) -> str:
    return (
        f"Summary of the earlier conversation with this user, up to {_time_label(section.latest_us)}:\n{section.text}"
    )


def _source_summary_text(source: str, section: _BackgroundSection) -> str:
    return f"Source {source} summary of earlier updates, up to {_time_label(section.latest_us)}:\n{section.text}"
