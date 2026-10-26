# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the conversation package's participant context store.

They use only the store's methods and what it renders, never its internal state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from xr_ai_conversation import CompactionPrompts, ContextManager, ConversationContextUpdate
from xr_ai_models import Capabilities, ChatResponse
from xr_ai_runtime import AgentRuntime, Topic

_MINUTE_US = 60_000_000
_NOON_US = 1_790_856_000_000_000
_NO_IDS: tuple[object | None, ...] = ()


class _LLM:
    """Answer every compaction request and let a test hold them back."""

    capabilities = Capabilities()

    def __init__(self) -> None:
        self.requests = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def chat(self, messages, **_kwargs):
        self.requests += 1
        self.started.set()
        await self.release.wait()
        return ChatResponse(content="Compacted facts.", reasoning=None, tool_calls=None, finish_reason="stop", raw={})


def _store(llm: _LLM | None = None, **overrides) -> ContextManager:
    options = {
        "summary_context_topics": ("status",),
        "incremental_context_topics": ("notes",),
        "dialogue_budget_bytes": 400,
        "topic_budget_bytes": 200,
        "compaction_input_budget_bytes": 12_000,
    }
    return ContextManager(llm or _LLM(), **{**options, **overrides})  # type: ignore[arg-type]


def _rendered(store: ContextManager, participant: str = "alice") -> str:
    return "\n".join(block.text for block in store.blocks(participant))


async def _eventually(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0)


def _note(text: str) -> ConversationContextUpdate:
    return ConversationContextUpdate(text=text)


def test_participant_without_context_renders_nothing():
    assert _store().blocks("alice") == ()


@pytest.mark.asyncio
async def test_recorded_exchange_renders_for_its_participant_only():
    store = _store()

    store.record_exchange("alice", "Remember copper.", "Copper noted.", timestamp_us=_NOON_US, exchange_id=None)

    assert "User: Remember copper.\nAssistant: “Copper noted.”" in _rendered(store, "alice")
    assert store.blocks("bob") == ()


@pytest.mark.asyncio
async def test_reconciled_history_windows_add_only_exchanges_not_already_known():
    store = _store()
    first = (("Hello.", "Hi.", _NOON_US), ("Bye.", "Goodbye.", _NOON_US + _MINUTE_US))
    overlapping = (("Bye.", "Goodbye.", _NOON_US + _MINUTE_US), ("Back.", "Welcome.", _NOON_US + 2 * _MINUTE_US))

    store.reconcile_history("alice", first, exchange_ids=_NO_IDS)
    store.reconcile_history("alice", first, exchange_ids=_NO_IDS)
    store.reconcile_history("alice", overlapping, exchange_ids=_NO_IDS)

    rendered = _rendered(store)
    assert [rendered.count(f"User: {user}") for user in ("Hello.", "Bye.", "Back.")] == [1, 1, 1]


@pytest.mark.asyncio
async def test_summary_topic_keeps_its_latest_state_and_incremental_topic_keeps_every_update():
    store = _store(topic_budget_bytes=2_000)

    for text in ("Timer at 4 minutes.", "Timer at 6 minutes."):
        await store.update("alice", "status", _note(text), mode="summary", timestamp_us=_NOON_US)
    for text in ("Three panels.", "One is interactive."):
        await store.update("alice", "notes", _note(text), mode="incremental", timestamp_us=_NOON_US)

    rendered = _rendered(store)
    assert "Timer at 6 minutes." in rendered
    assert "Timer at 4 minutes." not in rendered
    assert "Three panels." in rendered
    assert "One is interactive." in rendered


@pytest.mark.asyncio
async def test_blocks_are_ordered_by_the_time_of_their_content():
    store = _store(topic_budget_bytes=2_000)

    store.record_exchange(
        "alice", "Latest question.", "Latest answer.", timestamp_us=_NOON_US + 3 * _MINUTE_US, exchange_id=None
    )
    await store.update(
        "alice", "status", _note("Middle state."), mode="summary", timestamp_us=_NOON_US + 2 * _MINUTE_US
    )
    await store.update(
        "alice", "notes", _note("Earliest note."), mode="incremental", timestamp_us=_NOON_US + _MINUTE_US
    )

    rendered = _rendered(store)
    assert rendered.index("Earliest note.") < rendered.index("Middle state.") < rendered.index("Latest question.")


@pytest.mark.asyncio
async def test_incremental_context_beyond_its_budget_is_compacted_in_the_background():
    llm = _LLM()
    store = _store(llm)
    older, newer = "A" * 80, "B" * 80

    await store.update("alice", "notes", _note(older), mode="incremental", timestamp_us=_NOON_US)
    await store.update("alice", "notes", _note(newer), mode="incremental", timestamp_us=_NOON_US + 1)
    await _eventually(lambda: "Compacted facts." in _rendered(store))

    rendered = _rendered(store)
    assert older not in rendered
    assert llm.requests >= 1


@pytest.mark.asyncio
async def test_a_stuck_compaction_does_not_block_updates_or_rendering():
    llm = _LLM()
    llm.release.clear()
    store = _store(llm)

    await store.update("alice", "notes", _note("A" * 80), mode="incremental", timestamp_us=_NOON_US)
    await store.update("alice", "notes", _note("B" * 80), mode="incremental", timestamp_us=_NOON_US + 1)
    await _eventually(llm.started.is_set)
    await store.update(
        "alice", "notes", _note("Arrived during compaction."), mode="incremental", timestamp_us=_NOON_US + 2
    )

    assert "Arrived during compaction." in _rendered(store)
    llm.release.set()
    await store.clear("alice")


@pytest.mark.asyncio
async def test_clearing_a_participant_forgets_its_context_even_while_compaction_is_running():
    llm = _LLM()
    llm.release.clear()
    store = _store(llm)
    await store.update("alice", "notes", _note("A" * 80), mode="incremental", timestamp_us=_NOON_US)
    await store.update("alice", "notes", _note("B" * 80), mode="incremental", timestamp_us=_NOON_US + 1)
    await store.update("bob", "status", _note("Bob keeps this."), mode="summary", timestamp_us=_NOON_US)
    await _eventually(llm.started.is_set)

    await store.clear("alice")

    assert store.blocks("alice") == ()
    assert "Bob keeps this." in _rendered(store, "bob")
    # Releasing the held request cannot restore what was cleared; later context starts fresh.
    llm.release.set()
    await store.update("alice", "notes", _note("Fresh note."), mode="incremental", timestamp_us=_NOON_US + 2)
    rendered = _rendered(store)
    assert "Fresh note." in rendered
    assert "A" * 80 not in rendered
    assert "Compacted facts." not in rendered


@pytest.mark.asyncio
async def test_close_forgets_all_context_and_drops_later_topic_updates():
    store = _store()
    await store.update("alice", "status", _note("Before close."), mode="summary", timestamp_us=_NOON_US)

    await store.close()
    await store.update("alice", "status", _note("After close."), mode="summary", timestamp_us=_NOON_US)

    assert store.blocks("alice") == ()


@pytest.mark.asyncio
async def test_close_drops_later_exchanges_and_history():
    llm = _LLM()
    store = _store(llm)
    await store.close()

    # Each batch is past the dialogue's compaction trigger, so a recorded one would schedule compaction.
    for index in range(4):
        store.record_exchange(
            "alice",
            f"Late question {index} " + "q" * 80,
            "Late answer.",
            timestamp_us=_NOON_US + index,
            exchange_id=None,
        )
    store.reconcile_history(
        "alice",
        tuple((f"Old question {index} " + "q" * 80, "Old answer.", _NOON_US + index) for index in range(4)),
        exchange_ids=_NO_IDS,
    )
    await asyncio.sleep(0)

    assert store.blocks("alice") == ()
    assert llm.requests == 0


@pytest.mark.asyncio
async def test_compaction_prompts_are_customizable_and_the_manager_adds_the_size_limit():
    llm = _LLM()
    seen: list[str] = []
    chat = llm.chat

    async def recording_chat(messages, **kwargs):
        seen.append(messages[0].content)
        return await chat(messages, **kwargs)

    llm.chat = recording_chat  # type: ignore[method-assign]
    prompts = CompactionPrompts(source="Keep only part numbers.", dialogue="Keep only names.", condense="Shorter.")
    store = _store(llm, compaction_prompts=prompts)

    await store.update("alice", "notes", _note("A" * 80), mode="incremental", timestamp_us=_NOON_US)
    await store.update("alice", "notes", _note("B" * 80), mode="incremental", timestamp_us=_NOON_US + 1)
    await _eventually(lambda: "Compacted facts." in _rendered(store))

    assert seen[0].startswith("Keep only part numbers.")
    assert "at most 50 bytes" in seen[0]


@pytest.mark.asyncio
async def test_topic_update_without_a_participant_is_dropped(caplog):
    store = _store()
    runtime = AgentRuntime()
    runtime.register("context", store.subscriber())

    async with runtime:
        await runtime.publish(Topic("status", ConversationContextUpdate), {"text": "Nobody's state."})
        await _eventually(lambda: "dropping context update without participant" in caplog.text)

    assert store.blocks("alice") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "mode", "message"),
    [
        ("unknown", "summary", "not a configured topic"),
        ("status", "incremental", "configured as summary"),
        ("notes", "summary", "configured as incremental"),
    ],
)
async def test_update_rejects_an_unconfigured_source_or_the_wrong_mode(source, mode, message):
    store = _store()

    with pytest.raises(ValueError, match=message):
        await store.update("alice", source, _note("Ignored."), mode=mode, timestamp_us=_NOON_US)

    assert store.blocks("alice") == ()


@pytest.mark.asyncio
async def test_oversized_summary_state_is_clipped_within_its_budget(caplog):
    store = _store(topic_budget_bytes=300)

    await store.update("alice", "status", _note("start " + "x" * 5_000 + " end"), mode="summary", timestamp_us=_NOON_US)

    (block,) = store.blocks("alice")
    assert len(block.text.encode()) <= 300
    assert "start" in block.text
    assert block.text.endswith("end")
    assert "clipping oversized context state" in caplog.text


@pytest.mark.asyncio
async def test_dialogue_beyond_its_budget_is_compacted_and_the_newest_exchange_stays_verbatim():
    llm = _LLM()
    store = _store(llm)

    for index in range(4):
        store.record_exchange(
            "alice", f"Question {index} " + "q" * 80, f"Answer {index}", timestamp_us=_NOON_US + index, exchange_id=None
        )
    await _eventually(lambda: "Compacted facts." in _rendered(store))

    rendered = _rendered(store)
    assert "Question 0" not in rendered
    assert "User: Question 3" in rendered


@pytest.mark.asyncio
async def test_subscriber_applies_each_runtime_topic_with_its_mode_per_participant():
    store = _store(topic_budget_bytes=2_000)
    status, notes = Topic("status", ConversationContextUpdate), Topic("notes", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("context", store.subscriber())

    async with runtime:
        await runtime.publish(status, {"text": "Timer at 4 minutes."}, participant_id="alice")
        await runtime.publish(status, {"text": "Timer at 6 minutes."}, participant_id="alice")
        await runtime.publish(notes, {"text": "First note."}, participant_id="alice")
        await runtime.publish(notes, {"text": "Second note."}, participant_id="alice")
        await runtime.publish(status, {"text": "Bob's timer."}, participant_id="bob")
        await _eventually(lambda: "Bob's timer." in _rendered(store, "bob"))
        await _eventually(lambda: "Second note." in _rendered(store, "alice"))

    alice = _rendered(store, "alice")
    assert "Timer at 6 minutes." in alice
    assert "Timer at 4 minutes." not in alice
    assert "First note." in alice
    assert "Bob's timer." not in alice


def test_subscriber_requires_a_configured_topic():
    store = _store(summary_context_topics=(), incremental_context_topics=())

    with pytest.raises(ValueError, match="at least one context topic"):
        store.subscriber()


@pytest.mark.parametrize(
    "options",
    [
        {"summary_context_topics": ("a", "a")},
        {"summary_context_topics": "status"},
        {"incremental_context_topics": ("status",)},
        {"incremental_context_topics": ("  ",)},
        {"topic_budget_bytes": {"status": 100}},
        {"topic_budget_bytes": 0},
        {"dialogue_budget_bytes": -1},
        {"compaction_input_budget_bytes": 100},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        _store(**options)
