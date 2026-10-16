# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private participant dialogue and bounded source-context storage."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import nemo_relay
from xr_ai_routing._history import _format_minute


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


@dataclass(slots=True)
class _ParticipantMemory:
    transcript: list[_TranscriptEntry] = field(default_factory=list)
    context_records: dict[str, list[_ContextRecord]] = field(default_factory=dict)
    backgrounds: dict[str, _BackgroundSection] = field(default_factory=dict)
    dialogue_background: _BackgroundSection = field(default_factory=lambda: _BackgroundSection("", -1))
    revisions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dialogue_revision: int = 0
    dialogue_job: asyncio.Task[bool] | None = None
    context_jobs: dict[str, asyncio.Task[bool]] = field(default_factory=dict)
    context_order: list[str] = field(default_factory=list)
    retired_jobs: set[asyncio.Task[bool]] = field(default_factory=set)


_Compactor = Callable[[str, str, Sequence[_TranscriptEntry]], Awaitable[str]]
_DialogueCompactor = Callable[[str, Sequence[_TranscriptEntry]], Awaitable[str]]
_logger = logging.getLogger(__name__)
_Exchange = tuple[str, str, int | None, object | None]


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
    """Retain exact turns and compact old entries independently by source.

    ``payload_budget_bytes`` bounds the prior background and transcript text
    supplied to one compaction request. It does not include instructions or
    transport framing and is not a model-token estimate. Callers provide it
    from their own model/configuration boundary.
    """

    def __init__(self) -> None:
        self._participants: dict[str, _ParticipantMemory] = {}

    def append(
        self,
        participant: str,
        user: str,
        assistant: str,
        *,
        timestamp_us: int | None = None,
        exchange_id: object | None = None,
    ) -> _TranscriptEntry:
        memory = self._participants.setdefault(participant, _ParticipantMemory())
        entry = _TranscriptEntry(len(memory.transcript), "dialogue", user, assistant, timestamp_us, exchange_id)
        memory.transcript.append(entry)
        return entry

    def reconcile_transcript(
        self,
        participant: str,
        exchanges: Sequence[tuple[str, str] | tuple[str, str, int | None]],
        *,
        exchange_ids: Sequence[object | None] = (),
    ) -> None:
        """Merge a front-end history window without replaying retained exchanges."""

        memory = self._participants.setdefault(participant, _ParticipantMemory())
        existing = tuple(
            (entry.user, entry.assistant, entry.timestamp_us, entry.exchange_id) for entry in memory.transcript
        )
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

        # FE history is a bounded recent window and can start after every turn
        # retained here. Match its overlap with our tail; a non-overlapping
        # window is new completed context, not a divergent transcript to drop.
        if _contains_exchange_window(existing, normalized):
            return
        overlap = min(len(existing), len(normalized))
        while overlap and not all(
            _same_exchange(left, right) for left, right in zip(existing[-overlap:], normalized[:overlap])
        ):
            overlap -= 1
        for user, assistant, timestamp_us, exchange_id in normalized[overlap:]:
            self.append(participant, user, assistant, timestamp_us=timestamp_us, exchange_id=exchange_id)

    def transcript(self, participant: str) -> tuple[_TranscriptEntry, ...]:
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
        """Compact an old immutable dialogue prefix when unprocessed history is large."""

        memory = self._participants.get(participant)
        if memory is None:
            return None
        current = memory.dialogue_job
        if current is not None and not current.done():
            return current
        previous = memory.dialogue_background
        pending = tuple(entry for entry in memory.transcript if entry.sequence > previous.through_sequence)
        pending_bytes = sum(entry.utf8_size for entry in pending)
        if pending_bytes <= trigger_budget_bytes:
            return None
        # The newest complete exchange remains verbatim for the next turn.
        # Compact only the immutable prefix before it.
        prefix = pending[:-1]
        if not prefix:
            return None
        previous_input = _summary_for_compaction(previous, time_label="exchange completion times")
        available_bytes = payload_budget_bytes - len(previous_input.encode("utf-8"))
        if available_bytes <= 0:
            _logger.error(
                "dialogue compaction deferred participant=%r summary_bytes=%d payload_budget_bytes=%d",
                participant,
                len(previous_input.encode("utf-8")),
                payload_budget_bytes,
            )
            return None
        try:
            batch = _oldest_batch(prefix, available_bytes)
        except ValueError:
            _logger.error(
                "dialogue exchange exceeds compaction payload budget participant=%r budget_bytes=%d",
                participant,
                available_bytes,
            )
            return None
        revision = memory.dialogue_revision
        task = asyncio.create_task(
            self._compact_dialogue(
                participant,
                revision,
                previous,
                previous_input,
                batch,
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
        summary_budget_bytes: int,
        compact: _DialogueCompactor,
    ) -> bool:
        summary = await compact(previous_input, batch)
        memory = self._participants.get(participant)
        # Commit only while the prior summary and compacted transcript prefix
        # still match; newer tail turns may remain uncompacted.
        if memory is None or memory.dialogue_revision != revision:
            return False
        if memory.dialogue_background != previous:
            return False
        transcript_prefix = tuple(memory.transcript[entry.sequence] for entry in batch)
        if transcript_prefix != batch:
            return False
        summary_size = len(summary.encode("utf-8"))
        if summary_size > summary_budget_bytes:
            _logger.error(
                "dropping oversized dialogue summary participant=%r size_bytes=%d budget_bytes=%d",
                participant,
                summary_size,
                summary_budget_bytes,
            )
            return False
        memory.dialogue_background = _merge_background(summary, batch[-1].sequence, previous, batch)
        return True

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

        The caller rejects a summary that exceeds its budget before calling
        this method. Incremental records that exceed the budget are dropped as
        whole records; raw conversation transcript entries remain untouched.
        """

        memory = self._participants.setdefault(participant, _ParticipantMemory())
        if mode not in {"summary", "incremental"}:
            raise ValueError("context mode must be 'incremental' or 'summary'")
        if mode == "summary" and _rendered_source_size(source, text, timestamp_us) > context_budget_bytes:
            self._cancel_source_jobs(memory, source)
            memory.revisions[source] += 1
            memory.backgrounds.pop(source, None)
            memory.context_records.pop(source, None)
            return False, True
        if mode == "incremental" and len(text.encode("utf-8")) > context_budget_bytes:
            return False, True
        if mode == "summary":
            self._cancel_source_jobs(memory, source)
            memory.revisions[source] += 1
            if source not in memory.context_order:
                memory.context_order.append(source)
            memory.backgrounds[source] = _BackgroundSection(text, -1, timestamp_us, timestamp_us, timestamp_us is None)
            memory.context_records.pop(source, None)
            return True, False
        records = memory.context_records.setdefault(source, [])
        original_count = len(records)
        if source not in memory.context_order:
            memory.context_order.append(source)
        sequence = len(memory.transcript)
        records.append(_ContextRecord(sequence, text, timestamp_us))
        # Keep complete updates only, dropping the oldest whole source updates.
        while records and _source_context_size(memory, source, records) > context_budget_bytes:
            records.pop(0)
        dropped = len(records) != original_count + 1
        if not records:
            memory.context_records.pop(source, None)
        return bool(records), dropped

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
        payload_budget_bytes: int,
        summary_budget_bytes: int | None = None,
        compact: _Compactor,
    ) -> asyncio.Task[bool] | None:
        """Compact one source's incremental updates without blocking delivery."""

        memory = self._participants.get(participant)
        if memory is None or not memory.context_records.get(source):
            return None
        current = memory.context_jobs.get(source)
        if current is not None and not current.done():
            return current
        records = tuple(memory.context_records[source])
        if summary_budget_bytes is None:
            summary_budget_bytes = payload_budget_bytes
        if summary_budget_bytes <= 0:
            raise ValueError("summary_budget_bytes must be positive")
        previous = memory.backgrounds.get(source, _BackgroundSection("", -1))
        previous_input = _summary_for_compaction(previous, time_label="publication times")
        payload_size = len(previous_input.encode("utf-8")) + sum(
            len(record.text.encode("utf-8"))
            + len(
                (_format_minute(record.timestamp_us) if record.timestamp_us is not None else "unknown").encode("utf-8")
            )
            for record in records
        )
        if payload_size > payload_budget_bytes:
            return None
        revision = memory.revisions[source]
        entries = tuple(
            _TranscriptEntry(record.sequence, source, record.text, "", record.timestamp_us) for record in records
        )
        task = asyncio.create_task(
            self._compact_context(
                participant,
                source,
                revision,
                records,
                previous,
                previous_input,
                entries,
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
        records: tuple[_ContextRecord, ...],
        previous: _BackgroundSection,
        previous_input: str,
        entries: tuple[_TranscriptEntry, ...],
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _Compactor,
    ) -> bool:
        summary = await compact(source, previous_input, entries)
        memory = self._participants.get(participant)
        # Source revisions and record prefixes make this commit conditional on
        # the exact snapshot passed to the compactor.
        if memory is None or memory.revisions[source] != revision:
            return False
        if memory.backgrounds.get(source, _BackgroundSection("", -1)) != previous:
            return False
        current_records = tuple(memory.context_records.get(source, ()))
        if current_records[: len(records)] != records:
            return False
        summary_size = len(summary.encode("utf-8"))
        if summary_size > summary_budget_bytes:
            memory.backgrounds.pop(source, None)
            memory.context_records.pop(source, None)
            _logger.error(
                "dropping oversized compacted context participant=%r source=%r size_bytes=%d budget_bytes=%d",
                participant,
                source,
                summary_size,
                summary_budget_bytes,
            )
            return False
        memory.backgrounds[source] = _merge_background(summary, records[-1].sequence, previous, records)
        tail = current_records[len(records) :]
        if tail:
            memory.context_records[source] = list(tail)
        else:
            memory.context_records.pop(source, None)
        return True

    async def clear(self, participant: str) -> None:
        memory = self._participants.pop(participant, None)
        if memory is None:
            return
        tasks = (
            (() if memory.dialogue_job is None else (memory.dialogue_job,))
            + tuple(memory.context_jobs.values())
            + tuple(memory.retired_jobs)
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
        memory = self._participants.get(participant)
        if memory is not None and memory.dialogue_job is task:
            memory.dialogue_job = None
        _log_task_error(participant, "dialogue", task)
        if memory is not None and not task.cancelled() and task.exception() is None and task.result():
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
        payload_budget_bytes: int,
        summary_budget_bytes: int,
        compact: _Compactor,
    ) -> None:
        memory = self._participants.get(participant)
        if memory is not None and memory.context_jobs.get(source) is task:
            memory.context_jobs.pop(source, None)
        _log_task_error(participant, source, task)
        if (
            memory is not None
            and not task.cancelled()
            and task.exception() is None
            and memory.context_records.get(source)
        ):
            self.schedule_context_compaction(
                participant,
                source,
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


def _log_task_error(participant: str, source: str, task: asyncio.Task[bool]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        _logger.error(
            "conversation context compaction failed participant=%r source=%r",
            participant,
            source,
            exc_info=(type(error), error, error.__traceback__),
        )


def _oldest_batch(entries: tuple[_TranscriptEntry, ...], input_budget_bytes: int) -> tuple[_TranscriptEntry, ...]:
    batch = []
    size = 0
    for entry in entries:
        if batch and size + entry.utf8_size > input_budget_bytes:
            break
        if not batch and entry.utf8_size > input_budget_bytes:
            raise ValueError("one completed exchange exceeds compaction payload budget")
        batch.append(entry)
        size += entry.utf8_size
        if size >= input_budget_bytes:
            break
    return tuple(batch)


def _source_context_size(
    memory: _ParticipantMemory,
    source: str,
    records: Sequence[_ContextRecord],
) -> int:
    """Size one rendered source section, including its publication-time label."""

    background = memory.backgrounds.get(source, _BackgroundSection("", -1))
    text = "\n".join(filter(None, (background.text, *(record.text for record in records))))
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
    return _rendered_source_size(source, text, recency=recency)


def _rendered_source_size(
    source: str,
    text: str,
    timestamp_us: int | None = None,
    *,
    recency: str | None = None,
) -> int:
    if recency is None:
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
