# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the per-participant voice floor in the private IO processor."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from xr_ai_voice._frames import ParticipantLeftFrame, _OutputDoneFrame
from xr_ai_voice._processors.io import _VoiceIOProcessor
from xr_ai_voice._runtime import _ResponseStream
from xr_ai_voice._types import VoiceQuery


class _Floor:
    """One IO processor whose output frames are captured."""

    def __init__(self, **options) -> None:
        self.frames: list[Frame] = []
        self.queries: list[VoiceQuery] = []
        self.done: list[_OutputDoneFrame] = []
        self.processor = _VoiceIOProcessor(self._input, **options)
        self.processor.push_frame = self._capture  # type: ignore[method-assign]
        # Outside a pipeline: an interruption must not try to restart a process task.
        self.processor._enable_direct_mode = True  # noqa: SLF001

    async def _input(self, query: VoiceQuery) -> None:
        self.queries.append(query)

    async def _capture(self, frame: Frame, _direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        self.frames.append(frame)
        if isinstance(frame, _OutputDoneFrame):
            self.done.append(frame)

    @property
    def texts(self) -> list[str]:
        return [frame.text for frame in self.frames if isinstance(frame, TextFrame)]

    @property
    def interruptions(self) -> list[str]:
        return [
            frame.transport_source for frame in self.frames if isinstance(frame, InterruptionFrame)
        ]

    async def settle(self) -> None:
        await asyncio.sleep(0.02)

    async def bot(self, started: bool, pid: str = "alice") -> None:
        frame = BotStartedSpeakingFrame() if started else BotStoppedSpeakingFrame()
        frame.transport_destination = pid
        await self.processor.process_frame(frame, FrameDirection.UPSTREAM)
        await self.settle()

    async def user(self, started: bool, pid: str = "alice") -> None:
        frame = UserStartedSpeakingFrame() if started else UserStoppedSpeakingFrame()
        frame.transport_source = pid
        await self.processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        await self.settle()

    async def played(self) -> None:
        """Return the oldest done marker, as the output transport does after its audio."""
        done = self.done.pop(0)
        await self.processor.process_frame(
            _OutputDoneFrame(pid=done.pid, output=done.output), FrameDirection.UPSTREAM,
        )
        await self.settle()


class _Held:
    """A reply stream that yields one chunk and then holds the floor."""

    def __init__(self, chunk: str = "held") -> None:
        self.chunk = chunk
        self.release = asyncio.Event()
        self.closed = False
        self._sent = False

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if not self._sent:
            self._sent = True
            return self.chunk
        await self.release.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _Combiner:
    def __init__(self, result: str | Exception = "combined") -> None:
        self.result = result
        self.calls: list[list[str]] = []

    async def __call__(self, texts: Sequence[str]) -> str:
        self.calls.append(list(texts))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_reply_for_another_request_supersedes_and_drops_old_output() -> None:
    floor = _Floor()
    held = _Held("first ")
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.processor.enqueue_reply("alice", "queued", request_id="r1")
    await floor.settle()

    await floor.processor.enqueue_reply("alice", "second", request_id="r2")
    await floor.settle()
    await floor.processor.enqueue_reply("alice", "late", request_id="r1")
    await floor.settle()

    assert held.closed
    assert floor.texts == ["first ", "second"]
    assert floor.interruptions == ["alice"]


async def test_replies_for_the_same_request_play_in_order_without_interruption() -> None:
    floor = _Floor()
    held = _Held("one ")
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.processor.enqueue_reply("alice", "two", request_id="r1")
    await floor.settle()
    assert floor.texts == ["one "]

    held.release.set()
    await floor.settle()
    assert floor.texts == ["one "]

    await floor.played()
    assert floor.texts == ["one ", "two"]
    assert floor.interruptions == []


async def test_new_query_supersedes_active_request_and_cancels_replies() -> None:
    floor = _Floor()
    held = _Held()
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.processor.enqueue_reply("alice", "queued", request_id="r1")
    await floor.settle()

    await floor.processor.enqueue_query("alice", "next question")
    await floor.settle()
    await floor.processor.enqueue_reply("alice", "stale", request_id="r1")
    await floor.processor.enqueue_reply("alice", "fresh", request_id="r2")
    await floor.settle()

    assert held.closed
    assert floor.queries[0].interrupted_output is True
    assert floor.texts == ["held", "fresh"]


async def test_updates_wait_for_the_bot_to_stop_and_are_combined_once() -> None:
    combiner = _Combiner("Tea is ready and the timer ended.")
    floor = _Floor(combiner=combiner)
    await floor.bot(started=True)

    await floor.processor.enqueue_update("alice", "Tea is ready.", source="producer")
    await floor.processor.enqueue_update("alice", "The timer ended.", source="producer", pts_us=5)
    await floor.settle()
    assert floor.texts == []

    await floor.bot(started=False)

    assert combiner.calls == [["Tea is ready.", "The timer ended."]]
    assert floor.texts == ["Tea is ready and the timer ended."]


async def test_withdraw_drops_only_that_producers_pending_updates() -> None:
    combiner = _Combiner("kept")
    floor = _Floor(combiner=combiner)
    await floor.bot(started=True)
    await floor.processor.enqueue_update("alice", "stale one", source="guide")
    await floor.processor.enqueue_update("alice", "from timer", source="timer")
    await floor.processor.enqueue_update("alice", "stale two", source="guide")
    await floor.processor.enqueue_update("bob", "bob guide", source="guide")

    await floor.processor.withdraw_updates("alice", source="guide")
    await floor.bot(started=False)

    assert floor.texts == ["bob guide", "from timer"]
    assert combiner.calls == []


async def test_withdraw_drops_queued_urgent_updates_but_not_the_one_in_flight() -> None:
    floor = _Floor()
    gate = asyncio.Event()
    push_text = floor.processor._push_text  # noqa: SLF001

    async def held_push(text: str, *, pid: str) -> None:
        await gate.wait()
        await push_text(text, pid=pid)

    floor.processor._push_text = held_push  # type: ignore[method-assign]  # noqa: SLF001
    await floor.processor.enqueue_update("alice", "playing", urgent=True, source="guide")
    await floor.processor.enqueue_update("alice", "queued guide", urgent=True, source="guide")
    await floor.processor.enqueue_update("alice", "queued timer", urgent=True, source="timer")
    await floor.settle()

    await floor.processor.withdraw_updates("alice", source="guide")
    gate.set()
    await floor.settle()
    await floor.played()

    assert floor.texts == ["playing", "queued timer"]


async def test_withdraw_does_not_cancel_a_playing_reply() -> None:
    floor = _Floor()
    held = _Held()
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.settle()

    await floor.processor.withdraw_updates("alice", source="guide")

    assert floor.texts == ["held"]
    assert not held.closed
    assert floor.interruptions == []


async def test_update_speaks_immediately_when_the_bot_is_silent() -> None:
    combiner = _Combiner()
    floor = _Floor(combiner=combiner)

    await floor.processor.enqueue_update("alice", "Tea is ready.", source="producer")
    await floor.settle()

    assert floor.texts == ["Tea is ready."]
    assert combiner.calls == []


async def test_updates_are_joined_without_a_combiner() -> None:
    floor = _Floor()
    await floor.bot(started=True)
    await floor.processor.enqueue_update("alice", "One.", source="producer")
    await floor.processor.enqueue_update("alice", "Two.", source="producer")
    await floor.bot(started=False)

    assert floor.texts == ["One. Two."]


async def test_combiner_failure_falls_back_to_joined_updates() -> None:
    combiner = _Combiner(RuntimeError("model unavailable"))
    floor = _Floor(combiner=combiner)
    await floor.bot(started=True)
    await floor.processor.enqueue_update("alice", "One.", source="producer")
    await floor.processor.enqueue_update("alice", "Two.", source="producer")
    await floor.bot(started=False)

    assert combiner.calls == [["One.", "Two."]]
    assert floor.texts == ["One. Two."]


async def test_bot_frames_are_passed_on_and_tracked_per_participant() -> None:
    floor = _Floor()
    await floor.bot(started=True, pid="alice")

    await floor.processor.enqueue_update("bob", "For Bob.", source="producer")
    await floor.settle()

    assert [type(frame) for frame in floor.frames if not isinstance(frame, TextFrame)][:1] == [
        BotStartedSpeakingFrame
    ]
    assert floor.texts == ["For Bob."]


async def test_reply_for_a_new_request_cuts_audio_of_a_finished_reply() -> None:
    floor = _Floor()
    await floor.processor.enqueue_reply("alice", "old answer", request_id="r1")
    await floor.settle()
    await floor.bot(started=True)
    assert floor.processor._inflight == {}  # noqa: SLF001

    await floor.processor.enqueue_reply("alice", "new answer", request_id="r2")
    await floor.settle()
    await floor.processor.enqueue_reply("alice", "stale", request_id="r1")
    await floor.settle()

    assert floor.interruptions == ["alice"]
    assert floor.texts == ["old answer", "new answer"]


async def test_reply_for_a_new_request_cuts_audio_still_being_synthesized() -> None:
    floor = _Floor()
    await floor.processor.enqueue_reply("alice", "old answer", request_id="r1")
    await floor.settle()
    assert floor.processor._inflight == {}  # noqa: SLF001

    await floor.processor.enqueue_reply("alice", "new answer", request_id="r2")
    await floor.settle()

    assert floor.interruptions == ["alice"]
    assert floor.texts == ["old answer", "new answer"]


async def test_playback_pause_does_not_release_the_floor() -> None:
    floor = _Floor()
    await floor.processor.enqueue_reply("alice", "first", request_id="r1")
    await floor.settle()
    await floor.bot(started=True)
    await floor.bot(started=False)

    await floor.processor.enqueue_reply("alice", "second", request_id="r2")
    await floor.settle()

    assert floor.interruptions == ["alice"]


async def test_done_marker_of_cut_output_does_not_release_the_new_reply() -> None:
    floor = _Floor()
    await floor.processor.enqueue_reply("alice", "first", request_id="r1")
    await floor.settle()
    await floor.processor.enqueue_reply("alice", "second", request_id="r2")
    await floor.settle()
    await floor.played()

    await floor.processor.enqueue_reply("alice", "third", request_id="r3")
    await floor.settle()

    assert floor.interruptions == ["alice", "alice"]


async def test_new_request_supersedes_the_previous_one_during_silence() -> None:
    floor = _Floor()
    for text, request_id in (("one", "r1"), ("two", "r2")):
        await floor.processor.enqueue_reply("alice", text, request_id=request_id)
        await floor.settle()
        await floor.played()

    await floor.processor.enqueue_reply("alice", "late", request_id="r1")
    await floor.settle()

    assert floor.texts == ["one", "two"]


async def test_reply_for_the_same_request_does_not_cut_playing_audio() -> None:
    floor = _Floor()
    await floor.processor.enqueue_reply("alice", "one", request_id="r1")
    await floor.settle()
    await floor.bot(started=True)

    await floor.processor.enqueue_reply("alice", "two", request_id="r1")
    await floor.settle()
    assert floor.texts == ["one"]

    await floor.played()
    assert floor.interruptions == []
    assert floor.texts == ["one", "two"]


async def test_reply_does_not_cut_a_playing_urgent_update() -> None:
    floor = _Floor()
    await floor.processor.enqueue_update("alice", "Fire alarm.", source="producer", urgent=True)
    await floor.settle()
    interruptions = floor.interruptions
    await floor.bot(started=True)

    await floor.processor.enqueue_reply("alice", "answer", request_id="r1")
    await floor.settle()
    await floor.played()

    assert floor.interruptions == interruptions
    assert floor.texts == ["Fire alarm.", "answer"]


async def test_reply_cuts_a_playing_update() -> None:
    floor = _Floor()
    await floor.processor.enqueue_update("alice", "Timer done.", source="producer")
    await floor.settle()
    await floor.bot(started=True)

    await floor.processor.enqueue_reply("alice", "answer", request_id="r1")
    await floor.settle()

    assert floor.interruptions == ["alice"]


async def test_urgent_update_cuts_a_reply_and_supersedes_its_request() -> None:
    floor = _Floor()
    held = _Held("answer ")
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.processor.enqueue_reply("alice", "queued", request_id="r1")
    await floor.settle()

    await floor.processor.enqueue_update("alice", "Fire alarm.", source="producer", urgent=True)
    await floor.settle()
    await floor.processor.enqueue_reply("alice", "late", request_id="r1")
    await floor.settle()

    assert held.closed
    assert floor.texts == ["answer ", "Fire alarm."]
    assert floor.interruptions == ["alice"]


async def test_urgent_updates_queue_in_order_and_a_reply_waits_for_them() -> None:
    floor = _Floor()
    gate = asyncio.Event()
    push_text = floor.processor._push_text  # noqa: SLF001

    async def held_push(text: str, *, pid: str) -> None:
        await gate.wait()
        await push_text(text, pid=pid)

    floor.processor._push_text = held_push  # type: ignore[method-assign]  # noqa: SLF001

    for text in ("one", "two", "three"):
        await floor.processor.enqueue_update("alice", text, source="producer", urgent=True)
    await floor.processor.enqueue_reply("alice", "answer", request_id="r1")
    await floor.settle()
    assert floor.texts == []

    gate.set()
    await floor.settle()
    for _ in range(3):
        await floor.played()

    assert floor.texts == ["one", "two", "three", "answer"]
    assert floor.interruptions == ["alice"]


async def test_urgent_update_does_not_cut_a_playing_urgent_update() -> None:
    floor = _Floor()
    await floor.processor.enqueue_update("alice", "Fire alarm.", source="producer", urgent=True)
    await floor.settle()
    await floor.bot(started=True)

    await floor.processor.enqueue_update("alice", "Leave now.", source="producer", urgent=True)
    await floor.settle()
    assert floor.texts == ["Fire alarm."]

    await floor.played()
    assert floor.interruptions == ["alice"]
    assert floor.texts == ["Fire alarm.", "Leave now."]


async def test_queued_reply_does_not_take_the_floor_from_a_playing_urgent_update() -> None:
    floor = _Floor()
    await floor.processor.enqueue_update("alice", "Fire alarm.", source="producer", urgent=True)
    await floor.settle()
    await floor.processor.enqueue_reply("alice", "answer", request_id="r1")
    await floor.processor.enqueue_update("alice", "Leave now.", source="producer", urgent=True)
    await floor.settle()
    await floor.played()
    await floor.played()

    assert floor.interruptions == ["alice"]
    assert floor.texts == ["Fire alarm.", "Leave now.", "answer"]


async def test_barge_in_cancels_replies_but_keeps_updates() -> None:
    floor = _Floor()
    held = _Held()
    await floor.bot(started=True)
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.processor.enqueue_update("alice", "Timer done.", source="producer")
    await floor.settle()

    barge_in = InterruptionFrame()
    barge_in.transport_source = "alice"
    await floor.processor.process_frame(barge_in, FrameDirection.UPSTREAM)
    await floor.processor.enqueue_reply("alice", "stale", request_id="r1")
    await floor.settle()
    assert held.closed
    assert floor.texts == ["held"]

    await floor.bot(started=False)

    assert floor.texts == ["held", "Timer done."]


async def test_pending_updates_are_bounded_and_drop_the_oldest() -> None:
    combiner = _Combiner()
    floor = _Floor(combiner=combiner)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        await floor.bot(started=True)
        for index in range(66):
            await floor.processor.enqueue_update("alice", f"update {index}", source="producer")
        await floor.bot(started=False)
    finally:
        logger.remove(sink_id)

    assert len(combiner.calls) == 1
    assert combiner.calls[0][0] == "update 2"
    assert len(combiner.calls[0]) == 64
    assert sum("voice update dropped" in message for message in messages) == 2


async def test_participant_left_clears_the_floor() -> None:
    floor = _Floor()
    held = _Held()
    await floor.bot(started=True)
    await floor.processor.enqueue_reply("alice", held, request_id="r1")
    await floor.processor.enqueue_update("alice", "pending", source="producer")
    await floor.settle()

    await floor.processor.process_frame(
        ParticipantLeftFrame(participant_id="alice"), FrameDirection.DOWNSTREAM
    )
    await floor.bot(started=False)
    await floor.processor.enqueue_reply("alice", "again", request_id="r1")
    await floor.settle()

    assert held.closed
    assert floor.texts == ["held", "again"]


async def test_stalled_stream_is_closed_after_the_idle_timeout() -> None:
    closed: list[_ResponseStream] = []
    stream = _ResponseStream(4, closed.append, 0.02)
    await stream.send("spoken")

    fragments = [fragment async for fragment in stream]

    assert fragments == ["spoken"]
    assert closed == [stream]
    assert stream.closed.is_set()


class _GatedCombiner:
    """Combine updates only once ``release`` is set, recording each batch."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls: list[list[str]] = []

    async def __call__(self, texts: Sequence[str]) -> str:
        self.calls.append(list(texts))
        await self.release.wait()
        return " + ".join(texts)


async def test_updates_wait_while_the_participant_is_speaking() -> None:
    floor = _Floor()
    await floor.user(started=True)
    await floor.processor.enqueue_update("alice", "Timer done.", source="producer")
    barge_in = InterruptionFrame()
    barge_in.transport_source = "alice"
    await floor.processor.process_frame(barge_in, FrameDirection.UPSTREAM)
    await floor.bot(started=False)
    assert floor.texts == []

    await floor.user(started=False)

    assert floor.texts == ["Timer done."]


async def test_update_cut_by_a_reply_is_spoken_again_after_it() -> None:
    floor = _Floor()
    await floor.processor.enqueue_update("alice", "Timer done.", source="producer")
    await floor.settle()

    await floor.processor.enqueue_reply("alice", "answer", request_id="r1")
    await floor.settle()
    await floor.played()  # The cut update's marker no longer owns the floor.
    assert floor.texts == ["Timer done.", "answer"]

    await floor.played()

    assert floor.interruptions == ["alice"]
    assert floor.texts == ["Timer done.", "answer", "Timer done."]


async def test_updates_cancelled_while_combining_are_kept() -> None:
    combiner = _GatedCombiner()
    floor = _Floor(combiner=combiner)
    await floor.bot(started=True)
    await floor.processor.enqueue_update("alice", "one", source="producer")
    await floor.processor.enqueue_update("alice", "two", source="producer")
    await floor.bot(started=False)
    await floor.processor.enqueue_reply("alice", "answer", request_id="r1")
    await floor.settle()
    combiner.release.set()
    await floor.played()

    assert combiner.calls == [["one", "two"], ["one", "two"]]
    assert floor.texts == ["answer", "one + two"]


async def test_withdraw_while_combining_respeaks_only_the_remaining_updates() -> None:
    combiner = _GatedCombiner()
    floor = _Floor(combiner=combiner)
    await floor.bot(started=True)
    await floor.processor.enqueue_update("alice", "guide notice", source="guide")
    await floor.processor.enqueue_update("alice", "timer notice", source="timer")
    await floor.bot(started=False)

    await floor.processor.withdraw_updates("alice", source="guide")
    combiner.release.set()
    await floor.settle()

    assert floor.texts == ["timer notice"]


async def test_cancel_while_closing_a_stream_still_releases_the_floor() -> None:
    floor = _Floor()
    closing = asyncio.Event()

    class SlowClose(_Held):
        async def __anext__(self) -> str:
            if not self._sent:
                self._sent = True
                return self.chunk
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closing.set()
            await asyncio.Event().wait()

    await floor.processor.enqueue_reply("alice", SlowClose("answer"), request_id="r1")
    await asyncio.wait_for(closing.wait(), 1)
    await floor.processor.enqueue_query("alice", "next question")
    await floor.settle()
    await floor.processor.enqueue_update("alice", "Timer done.", source="producer")
    await floor.played()

    assert floor.texts == ["answer", "Timer done."]
