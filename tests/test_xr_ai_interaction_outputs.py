# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the producer-side response output helper."""

from __future__ import annotations

import asyncio

import pytest
from xr_ai_interaction import Reply, ResponseOutput, ResponseOutputs

TOPIC = "voice.output"


class _ClosedError(RuntimeError):
    pass


class _Publisher:
    def __init__(self, *, fail_on: int | None = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[object, ResponseOutput, str, dict]] = []
        self._fail_on = fail_on
        self._error = error or RuntimeError("publish failed")

    async def publish(self, topic, message, *, participant_id, **kwargs) -> None:
        if self._fail_on is not None and len(self.calls) + 1 == self._fail_on:
            raise self._error
        self.calls.append((topic, message, participant_id, kwargs))

    @property
    def outputs(self) -> list[ResponseOutput]:
        return [message for _topic, message, _pid, _kwargs in self.calls]


async def test_update_publishes_one_update_and_skips_blank_text() -> None:
    publisher = _Publisher()
    outputs = ResponseOutputs(publisher, TOPIC)

    await outputs.update("alice", "Timer done.", urgent=True, timestamp_us=7)
    await outputs.update("alice", "   ")

    assert publisher.calls == [
        (
            TOPIC,
            ResponseOutput(kind="update", text="Timer done.", urgent=True, timestamp_us=7),
            "alice",
            {},
        )
    ]


async def test_withdraw_publishes_one_withdraw_with_the_source() -> None:
    publisher = _Publisher()

    await ResponseOutputs(publisher, TOPIC, source="guide").withdraw("alice")

    assert publisher.calls == [(TOPIC, ResponseOutput(kind="withdraw"), "alice", {"source": "guide"})]


async def test_source_is_passed_only_when_set() -> None:
    publisher = _Publisher()

    await ResponseOutputs(publisher, TOPIC, source="timer").update("alice", "Done.")

    assert publisher.calls[0][3] == {"source": "timer"}


async def test_reply_streams_fragments_then_publishes_the_terminator() -> None:
    publisher = _Publisher()
    outputs = ResponseOutputs(publisher, TOPIC)

    async with outputs.reply("alice", request_id="r1", timestamp_us=9) as reply:
        assert isinstance(reply, Reply)
        await reply.send("Hello ")
        await reply.send("world.")
        assert reply.delivered is False

    fragments = publisher.outputs
    assert [(o.text, o.final) for o in fragments] == [("Hello ", False), ("world.", False), ("", True)]
    assert {o.kind for o in fragments} == {"reply"}
    assert {o.request_id for o in fragments} == {"r1"}
    assert {o.timestamp_us for o in fragments} == {9}
    assert len({o.response_id for o in fragments}) == 1 and fragments[0].response_id
    assert reply.delivered is True


async def test_each_reply_has_its_own_response_id() -> None:
    publisher = _Publisher()
    outputs = ResponseOutputs(publisher, TOPIC)

    for _ in range(2):
        async with outputs.reply("alice", request_id="r1") as reply:
            await reply.send("text")

    ids = {o.response_id for o in publisher.outputs}
    assert len(ids) == 2


async def test_started_is_awaited_once_before_the_first_fragment_and_blank_lead_is_skipped() -> None:
    publisher = _Publisher()
    events: list[str] = []

    async def started() -> None:
        events.append(f"started after {len(publisher.calls)} publishes")

    async with ResponseOutputs(publisher, TOPIC).reply(
        "alice", request_id="r1", started=started
    ) as reply:
        await reply.send("  ")
        await reply.send("")
        assert events == []
        await reply.send(" Hi")
        await reply.send("there")

    assert events == ["started after 0 publishes"]
    assert [o.text for o in publisher.outputs] == [" Hi", "there", ""]


async def test_reply_without_fragments_publishes_nothing() -> None:
    publisher = _Publisher()

    async with ResponseOutputs(publisher, TOPIC).reply("alice", request_id="r1") as reply:
        await reply.send("   ")

    assert publisher.calls == []
    assert reply.delivered is False


async def test_exception_in_body_still_publishes_the_terminator() -> None:
    publisher = _Publisher()
    reply = ResponseOutputs(publisher, TOPIC).reply("alice", request_id="r1")

    with pytest.raises(ValueError, match="boom"):
        async with reply:
            await reply.send("partial")
            raise ValueError("boom")

    assert [o.final for o in publisher.outputs] == [False, True]
    assert reply.delivered is True


async def test_cancellation_publishes_the_terminator_and_blocks_later_fragments() -> None:
    publisher = _Publisher()
    reached_send = asyncio.Event()
    reply = ResponseOutputs(publisher, TOPIC).reply("alice", request_id="r1")

    async def produce() -> None:
        async with reply:
            await reply.send("partial")
            reached_send.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # A model client may swallow cancellation; the reply must still not publish.
                await reply.send("must not be published")

    task = asyncio.create_task(produce())
    await reached_send.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [o.text for o in publisher.outputs] == ["partial", ""]
    assert reply.delivered is True


async def test_failed_fragment_publish_is_not_closed() -> None:
    publisher = _Publisher(fail_on=2)
    reply = ResponseOutputs(publisher, TOPIC).reply("alice", request_id="r1")

    with pytest.raises(RuntimeError, match="publish failed"):
        async with reply:
            await reply.send("one")
            await reply.send("two")

    assert [o.text for o in publisher.outputs] == ["one"]
    assert reply.delivered is False


async def test_closed_errors_from_the_terminator_are_suppressed() -> None:
    publisher = _Publisher(fail_on=2, error=_ClosedError("closed"))
    outputs = ResponseOutputs(publisher, TOPIC, closed_errors=(_ClosedError,))

    async with outputs.reply("alice", request_id="r1") as reply:
        await reply.send("one")

    assert reply.delivered is False
    assert [o.text for o in publisher.outputs] == ["one"]


async def test_other_terminator_errors_propagate() -> None:
    publisher = _Publisher(fail_on=2)
    reply = ResponseOutputs(publisher, TOPIC, closed_errors=(_ClosedError,)).reply(
        "alice", request_id="r1"
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        async with reply:
            await reply.send("one")

    assert reply.delivered is False


async def test_send_after_exit_is_a_caller_error() -> None:
    publisher = _Publisher()
    reply = ResponseOutputs(publisher, TOPIC).reply("alice", request_id="r1")
    async with reply:
        await reply.send("one")

    with pytest.raises(RuntimeError, match="closed"):
        await reply.send("two")


async def test_cancellation_during_a_fragment_publish_still_publishes_the_terminator() -> None:
    class _CancelOnce(_Publisher):
        async def publish(self, topic, message, *, participant_id, **kwargs) -> None:
            if len(self.calls) == 1 and not message.final:
                raise asyncio.CancelledError
            await super().publish(topic, message, participant_id=participant_id, **kwargs)

    publisher = _CancelOnce()
    reply = ResponseOutputs(publisher, TOPIC).reply("alice", request_id="r1")

    with pytest.raises(asyncio.CancelledError):
        async with reply:
            await reply.send("one")
            await reply.send("two")

    assert [o.text for o in publisher.outputs] == ["one", ""]
    assert reply.delivered is True
