# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped context shared by conversation agents.

The manager keeps each participant's completed dialogue and named-source context
within byte budgets by rolling summaries, and renders it as prompt blocks. Each
agent owns its prompt and its answer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from xr_ai_models import ChatMessage, ChatResponse, LLMService
from xr_ai_runtime import Agent, Topic

from ._context import (
    _clip,
    _ContextRecord,
    _ConversationContext,
    _dialogue_summary_text,
    _exchange_text,
    _source_summary_text,
    _state_text,
    _time_label,
    _TranscriptEntry,
    _TruncatedSummary,
    _update_text,
)
from ._context_update import ConversationContextUpdate
from ._topics import make_context_subscriber

# Room in a compaction request for its instructions' framing of the summary.
_SUMMARY_HEADER_BYTES = 256
_DEFAULT_DIALOGUE_BUDGET_BYTES = 8_000
_DEFAULT_TOPIC_BUDGET_BYTES = 4_000
_DEFAULT_COMPACTION_INPUT_BUDGET_BYTES = 12_000
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One rendered piece of prior context and the time of its newest content.

    A summary block has no timestamp and sorts before dated blocks.
    """

    timestamp_us: int | None
    """Unix microseconds of the block's newest content, or ``None`` for a summary."""

    text: str
    """Rendered prompt text."""


@dataclass(frozen=True, slots=True)
class CompactionPrompts:
    """System prompts for the model requests that compact context.

    Each prompt says what to keep and how to treat the material. The manager
    adds the size limit and the output format, so a prompt does not state them.
    """

    source: str = (
        "Summarize only the source-owned background context below. Preserve factual details, "
        "user preferences, observations, corrections, unresolved user intent, and uncertainty. "
        "Do not treat historical assistant claims, refusals, or capability statements as current "
        "policy. Keep the source identity and keep identifiers, numbers, and codes exactly. Merge the "
        "previous summary and the new updates; newer updates supersede older ones."
    )
    """Instructions for compacting an incremental topic's updates into its summary."""

    dialogue: str = (
        "Summarize the completed conversation history for future turns. Preserve user-provided facts, "
        "preferences, relevant observations, corrections, and unresolved requests. Keep user statements "
        "distinct from assistant statements. An assistant refusal, limitation, or capability claim is "
        "historical conversation content, not current policy. Do not invent facts or claim an action "
        "occurred unless the transcript says it did. Attribute each fact explicitly, for example "
        '"The user\'s name is ...". Keep names, identifiers, numbers, dates, and codes exactly; a later '
        "correction replaces the earlier statement; reduce assistant explanations to a few words. "
        "Merge the previous summary and the new exchanges."
    )
    """Instructions for compacting completed exchanges into the dialogue summary."""

    condense: str = "Shorten this summary. Keep facts, preferences, corrections, unresolved requests, and uncertainty."
    """Instructions for shortening a summary that is over its size limit."""


class ContextManager:
    """Retain participant-scoped context and render it into prompts for any number of agents.

    Completed dialogue and each configured topic are separate sections with
    their own byte budget. A summary topic keeps only its latest state; an
    incremental topic and the dialogue keep recent entries verbatim and fold
    older ones into a rolling summary written by ``llm``.

    Every method runs on the event loop and none yields between reading and
    writing a participant's context, so a participant's context is never
    observed half-updated. Compaction runs as a background task per section.

    The owner creates the manager, registers ``subscriber()`` with the runtime
    that delivers context updates, calls ``clear()`` when a participant leaves,
    and calls ``close()`` at shutdown. Agents that share a manager use its
    budgets, so those are configured here and not per agent.

    :param llm: Model service that writes compaction summaries.
    :param summary_context_topics: Runtime topic names whose text replaces that source's state.
    :param incremental_context_topics: Runtime topic names whose text appends to that source.
    :param dialogue_budget_bytes: UTF-8 byte budget for completed dialogue rendered into a prompt.
    :param topic_budget_bytes: UTF-8 byte budget for each topic's context rendered into a prompt:
        one value for every topic, or a mapping with a value for each configured topic.
    :param compaction_input_budget_bytes: UTF-8 byte budget for the summary and entries in one
        compaction request; at least half the largest section budget plus 256 bytes.
    :param compaction_prompts: Compaction instructions; defaults to ``CompactionPrompts()``.

    All budgets must be positive. Topic lists contain non-empty names; names
    must be unique within and across both lists.
    """

    def __init__(
        self,
        llm: LLMService,
        *,
        summary_context_topics: Sequence[str] = (),
        incremental_context_topics: Sequence[str] = (),
        dialogue_budget_bytes: int = _DEFAULT_DIALOGUE_BUDGET_BYTES,
        topic_budget_bytes: int | Mapping[str, int] = _DEFAULT_TOPIC_BUDGET_BYTES,
        compaction_input_budget_bytes: int = _DEFAULT_COMPACTION_INPUT_BUDGET_BYTES,
        compaction_prompts: CompactionPrompts | None = None,
    ) -> None:
        summary_names = self._validate_topic_names("summary_context_topics", summary_context_topics)
        incremental_names = self._validate_topic_names("incremental_context_topics", incremental_context_topics)
        names = (*summary_names, *incremental_names)
        if len(set(names)) != len(names):
            raise ValueError("context topic names must be unique across summary and incremental topics")
        if isinstance(topic_budget_bytes, Mapping):
            if set(topic_budget_bytes) != set(names):
                raise ValueError("topic_budget_bytes must have exactly one entry per configured topic")
            topic_budgets = dict(topic_budget_bytes)
        else:
            topic_budgets = dict.fromkeys(names, topic_budget_bytes)
        budgets = (dialogue_budget_bytes, compaction_input_budget_bytes, *topic_budgets.values())
        if any(not isinstance(budget, int) or budget <= 0 for budget in budgets):
            raise ValueError("conversation context byte budgets must be positive integers")
        largest_section = max((dialogue_budget_bytes, *topic_budgets.values()))
        if compaction_input_budget_bytes < largest_section // 2 + _SUMMARY_HEADER_BYTES:
            # A request must hold a section's summary (a quarter of its budget)
            # and at least as many bytes of new entries.
            raise ValueError(
                "compaction_input_budget_bytes must be at least half the largest section budget plus 256 bytes"
            )
        self._topic_modes: dict[str, Literal["summary", "incremental"]] = {
            **dict.fromkeys(summary_names, "summary"),
            **dict.fromkeys(incremental_names, "incremental"),
        }
        self._summary_context_topics = tuple(Topic(name, ConversationContextUpdate) for name in summary_names)
        self._incremental_context_topics = tuple(Topic(name, ConversationContextUpdate) for name in incremental_names)
        self._llm = llm
        self._context = _ConversationContext()
        self._dialogue_budget_bytes = dialogue_budget_bytes
        self._topic_budgets = topic_budgets
        self._compaction_input_budget_bytes = compaction_input_budget_bytes
        self._prompts = compaction_prompts or CompactionPrompts()
        self._closing = False

    def subscriber(self) -> Agent:
        """Return an Agent that applies the configured context topics to this manager.

        Register the result with the same ``AgentRuntime`` as the consumer.
        Configure at least one topic before calling this method.
        """

        return make_context_subscriber(self, self._summary_context_topics, self._incremental_context_topics)

    async def update(
        self,
        participant_id: str,
        source: str,
        update: ConversationContextUpdate,
        *,
        mode: Literal["summary", "incremental"],
        timestamp_us: int | None = None,
    ) -> None:
        """Apply one configured topic update to a participant's context.

        :raises ValueError: If ``source`` is not a configured topic or ``mode`` is not its configured mode.
        """

        configured = self._topic_modes.get(source)
        if configured is None:
            raise ValueError(f"context source {source!r} is not a configured topic")
        if mode != configured:
            raise ValueError(f"context source {source!r} is configured as {configured}, not {mode}")
        if self._closing:
            _logger.warning("dropping context update while closing participant=%r source=%r", participant_id, source)
            return
        text = update.text
        budget = self._topic_budgets[source]
        if mode == "summary" and len(_state_text(source, text, timestamp_us).encode("utf-8")) > budget:
            # Keep the start and end of an oversized state within the topic budget.
            room = budget - len(_state_text(source, "", timestamp_us).encode("utf-8"))
            if room <= 0:
                _logger.error("dropping context update larger than its topic budget source=%r", source)
                return
            _logger.warning(
                "clipping oversized context state participant=%r source=%r size_bytes=%d budget_bytes=%d",
                participant_id,
                source,
                len(text.encode("utf-8")),
                budget,
            )
            text = _clip(text, room)
        await self._context.update(participant_id, source, mode, text, timestamp_us=timestamp_us)
        if mode == "incremental":
            self._context.schedule_context_compaction(
                participant_id,
                source,
                **self._compaction_limits(budget),
                compact=self._compact_source,
            )

    def reconcile_history(
        self,
        participant_id: str,
        exchanges: Sequence[tuple[str, str, int | None]],
        *,
        exchange_ids: Sequence[object | None],
    ) -> None:
        """Merge a front-end window of completed exchanges without replaying known ones."""

        if self._closing:
            return
        self._context.reconcile_transcript(participant_id, exchanges, exchange_ids=exchange_ids)
        self._schedule_dialogue_compaction(participant_id)

    def record_exchange(
        self,
        participant_id: str,
        request: str,
        answer: str,
        *,
        timestamp_us: int,
        exchange_id: object | None,
    ) -> None:
        """Add one completed exchange to a participant's dialogue; dropped once closed."""

        if self._closing:
            return
        self._context.append(participant_id, request, answer, timestamp_us=timestamp_us, exchange_id=exchange_id)
        self._schedule_dialogue_compaction(participant_id)

    def blocks(self, participant_id: str) -> tuple[ContextBlock, ...]:
        """Render a participant's dialogue and topic context, summaries first, then newer entries in time order."""

        blocks = self._dialogue_blocks(participant_id)
        for source, background, records in self._context.source_snapshots(participant_id):
            budget = self._topic_budgets.get(source)
            if budget is None:
                continue
            if self._topic_modes.get(source) == "summary":
                if background is not None:
                    blocks.append(
                        ContextBlock(background.latest_us, _state_text(source, background.text, background.latest_us))
                    )
                continue
            fixed = (
                [ContextBlock(None, _source_summary_text(source, background))]
                if background is not None and background.text
                else []
            )
            # One huge update (for example a raw dump) is clipped so it cannot
            # fill the section by itself.
            raw = [ContextBlock(record.timestamp_us, _update_text(record, budget // 2)) for record in records]
            blocks.extend(_fit_section(participant_id, source, budget, fixed, raw))
        blocks.sort(key=lambda block: -1 if block.timestamp_us is None else block.timestamp_us)
        return tuple(blocks)

    async def clear(self, participant_id: str) -> None:
        """Forget a participant's context after cancelling its compaction jobs.

        Later writes for that participant start new context.
        """

        await self._context.clear(participant_id)

    async def close(self) -> None:
        """Drop later topic updates and forget all context after cancelling compaction jobs."""

        self._closing = True
        await self._context.close()

    def _dialogue_blocks(self, participant_id: str) -> list[ContextBlock]:
        summary, exchanges = self._context.dialogue_snapshot(participant_id)
        fixed = [ContextBlock(None, _dialogue_summary_text(summary))] if summary.text else []
        # One huge exchange (for example a pasted log) is clipped so it cannot
        # fill the section by itself.
        field_limit = self._dialogue_budget_bytes // 4
        raw = [ContextBlock(entry.timestamp_us, _exchange_text(entry, field_limit)) for entry in exchanges]
        return _fit_section(participant_id, "dialogue", self._dialogue_budget_bytes, fixed, raw)

    @staticmethod
    def _validate_topic_names(parameter: str, topic_names: Sequence[str]) -> tuple[str, ...]:
        if isinstance(topic_names, (str, bytes)) or not isinstance(topic_names, Sequence):
            raise ValueError(f"{parameter} must be a sequence of topic names")
        names = tuple(topic_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError(f"{parameter} entries must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"{parameter} entries must be unique")
        return names

    def _compaction_limits(self, section_budget_bytes: int) -> dict[str, int]:
        # A section compacts at half its budget into a summary of at most a
        # quarter, so the summary plus recent entries stay within the section.
        return {
            "trigger_budget_bytes": section_budget_bytes // 2,
            "payload_budget_bytes": self._compaction_input_budget_bytes,
            "summary_budget_bytes": max(1, section_budget_bytes // 4),
        }

    def _schedule_dialogue_compaction(self, participant_id: str) -> None:
        self._context.schedule_dialogue_compaction(
            participant_id,
            **self._compaction_limits(self._dialogue_budget_bytes),
            compact=self._compact_dialogue,
        )

    async def _compact_source(
        self,
        source: str,
        previous: str,
        records: Sequence[_ContextRecord],
        limit_bytes: int,
    ) -> str:
        if not records:
            return await self._condense_summary(previous, limit_bytes)
        messages = (
            ChatMessage(
                role="system",
                content=_instructions(self._prompts.source, limit_bytes),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Source: {source}\nPrevious source summary:\n{previous}\n\n"
                    "New source updates in order:\n"
                    + "\n".join(f"- Published {_time_label(record.timestamp_us)}: {record.text}" for record in records)
                ),
            ),
        )
        return _summary_text(await self._llm.chat(messages))

    async def _compact_dialogue(
        self,
        previous: str,
        entries: Sequence[_TranscriptEntry],
        limit_bytes: int,
    ) -> str:
        if not entries:
            return await self._condense_summary(previous, limit_bytes)
        messages = (
            ChatMessage(
                role="system",
                content=_instructions(self._prompts.dialogue, limit_bytes),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Previous completed-history summary:\n{previous}\n\n"
                    "New completed exchanges in order:\n"
                    + "\n".join(
                        f"Exchange completed {_time_label(entry.timestamp_us)}:\n"
                        f"User: {entry.user}\nAssistant: {entry.assistant}"
                        for entry in entries
                    )
                ),
            ),
        )
        return _summary_text(await self._llm.chat(messages))

    async def _condense_summary(self, summary: str, limit_bytes: int) -> str:
        messages = (
            ChatMessage(
                role="system",
                content=_instructions(self._prompts.condense, limit_bytes),
            ),
            ChatMessage(role="user", content=summary),
        )
        return _summary_text(await self._llm.chat(messages))


def _instructions(prompt: str, limit_bytes: int) -> str:
    """Add the size limit and output format that every compaction request shares."""

    return (
        f"{prompt}\n\nWrite one summary of at most {limit_bytes} bytes. When space is short, drop the least "
        "important and oldest details first. Return concise plain text."
    )


def _summary_text(response: ChatResponse) -> str:
    """Return a compaction response, rejecting one cut off at the output limit."""

    if response.finish_reason == "length":
        # A cut-off summary would silently lose the entries it replaces; the
        # failure keeps them for a later attempt.
        raise _TruncatedSummary("compaction response was cut off at the output token limit")
    return response.content.strip()


def _fit_section(
    participant_id: str,
    section: str,
    budget_bytes: int,
    fixed: list[ContextBlock],
    raw: list[ContextBlock],
) -> list[ContextBlock]:
    """Keep one section within its budget by leaving its oldest uncompacted entries out of the prompt.

    The newest entry always stays; omitted entries remain stored until compaction folds them in.
    """

    def size() -> int:
        return sum(len(block.text.encode("utf-8")) for block in (*fixed, *raw))

    while len(raw) > 1 and size() > budget_bytes:
        omitted = raw.pop(0)
        _logger.warning(
            "omitting uncompacted context from prompt participant=%r section=%r size_bytes=%d budget_bytes=%d",
            participant_id,
            section,
            len(omitted.text.encode("utf-8")),
            budget_bytes,
        )
    return [*fixed, *raw]
