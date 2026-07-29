# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``StreamingTtsProcessor`` — per-participant sentence-batched parallel TTS.

Consumes ``TextFrame``s (assistant output, one per chunk/token) and emits
``OutputAudioRawFrame``s. Buffers text per participant until a sentence
boundary, synthesizes each sentence in parallel, then streams the WAVs out in
order so each participant's playback stays monotonic.

All streaming state — pending text, the synth/order queue, and the sender task
— is keyed by participant id, so concurrent participants never share a buffer
(which would interleave their words) or a sender (which would misroute audio).

An ``InterruptionFrame`` cancels only the interrupting participant's in-flight
synthesis and flushes only their hub audio (``transport_source`` carries the
pid); a frame with no pid falls back to draining every participant. A pipeline
``EndFrame``/``CancelFrame`` tears down all sender tasks.

Every synthesized WAV is offered to ``VoiceGate.observe_tts_wav`` so the gate's
listening chime can lazily build at the right sample rate. When constructed with
a non-empty ``text_topic`` and a ``transport``, the processor also echoes each
assistant turn's full assembled response on the data channel under that topic.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_agent import DataMessage
from xr_ai_models import TTSService
from xr_ai_voicegate import VoiceGate

from .._audio import wav_to_output_frames
from .._frames import AssistantResponseEndFrame

if TYPE_CHECKING:
    from .._transport import HubVoiceTransport


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A sentence-final char optionally followed by closing punctuation (quote,
# paren, bracket) and trailing whitespace, e.g. ``... looking at?"``. A plain
# ``endswith((".", "!", "?"))`` would miss the trailing quote and leave the tail
# in the buffer to concatenate onto the next turn.
_TRAILING_SENTENCE_END = re.compile(r"""[.!?]["')\]]*\s*$""")


class _TtsPidState:
    """Per-participant streaming state: pending text + its ordered sender."""

    __slots__ = ("pending", "sender_task", "sender_queue", "synth_seq")

    def __init__(self) -> None:
        self.pending: str = ""
        self.sender_task: asyncio.Task | None = None
        self.sender_queue: asyncio.Queue | None = None
        self.synth_seq: int = 0


class StreamingTtsProcessor(FrameProcessor):
    """Per-participant sentence-batched parallel TTS at the pipeline tail.

    ``transport`` and ``text_topic`` are optional; when both are supplied (and
    the topic is non-empty), the processor emits one ``send_return_data`` per
    :class:`AssistantResponseEndFrame` so the client receives the full assembled
    reply on the data channel. Leaving them out (or passing an empty topic)
    disables the echo — used by samples whose assistant already sends its own
    per-turn data response.
    """

    def __init__(
        self,
        *,
        tts: TTSService,
        voice_gate: VoiceGate,
        transport: "HubVoiceTransport | None" = None,
        text_topic: str = "",
    ) -> None:
        super().__init__()
        self._tts        = tts
        self._voice_gate = voice_gate
        self._transport  = transport
        self._text_topic = text_topic
        # Per-participant streaming state, keyed by pid.
        self._by_pid: dict[str, _TtsPidState] = {}

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                await self._drain_on_interrupt(pid)
            else:
                # No pid on the frame — drain every participant (legacy fallback).
                await self._drain_all_on_interrupt()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            # Normal pipeline shutdown: cancel/await every participant's sender
            # task so no TTS background work is retained.
            await self._shutdown_all()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, AssistantResponseEndFrame):
            await self._handle_response_end(frame)
            # Forward the marker so any tail processor / sink that tracks turn
            # boundaries still sees it.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            await self._handle_text(frame)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    def _state(self, pid: str) -> _TtsPidState:
        st = self._by_pid.get(pid)
        if st is None:
            st = _TtsPidState()
            self._by_pid[pid] = st
        return st

    def _ensure_sender(self, pid: str) -> asyncio.Queue:
        """Spin up ``pid``'s ordered-sender task lazily on first sentence."""
        st = self._state(pid)
        if st.sender_queue is None or st.sender_task is None or st.sender_task.done():
            st.sender_queue = asyncio.Queue()
            st.sender_task  = asyncio.create_task(
                self._sender_loop(st.sender_queue),
                name=f"tts-sender-{pid}",
            )
        return st.sender_queue

    async def _handle_text(self, frame: TextFrame) -> None:
        if not frame.text:
            return
        # transport_destination carries the addressed participant; text with no
        # destination buffers under "" (single-participant / fallback routing).
        pid = frame.transport_destination or ""
        self._state(pid).pending += frame.text
        await self._flush_complete_sentences(pid)

    async def _handle_response_end(self, frame: AssistantResponseEndFrame) -> None:
        """Flush ``frame.pid``'s trailing pending text, then send the data echo.

        The assistant may finish a turn with text that has no sentence-final
        punctuation (e.g. an aborted partial answer); the boundary regex would
        leave that fragment buffered forever. End-of-response flushes it so the
        user hears the tail of the reply.
        """
        st = self._by_pid.get(frame.pid)
        if st is not None and st.pending.strip():
            sentence = st.pending.strip()
            st.pending = ""
            await self._dispatch_sentence(sentence, pid=frame.pid)

        if not self._text_topic or self._transport is None:
            return
        if not frame.text:
            return
        logger.info(
            "data text echo pid={!r} topic={!r} len={}",
            frame.pid, self._text_topic, len(frame.text),
        )
        try:
            await self._transport.send_return_data(DataMessage(
                participant_id = frame.pid,
                topic          = self._text_topic,
                pts_us         = frame.pts_us,
                data           = frame.text.encode(),
            ))
        except Exception:
            logger.exception(
                "send_return_data failed pid={!r} topic={!r}",
                frame.pid, self._text_topic,
            )

    async def _flush_complete_sentences(self, pid: str) -> None:
        """Drain every complete sentence in ``pid``'s pending buffer, leaving a
        trailing fragment in place until more text arrives. A buffer already
        ending in sentence-final punctuation is flushed in one shot — covers the
        assistant-returns-a-complete-string case with no trailing whitespace."""
        st = self._state(pid)
        while True:
            m = _SENTENCE_END.search(st.pending)
            if m is None:
                break
            sentence  = st.pending[: m.end()].strip()
            st.pending = st.pending[m.end() :]
            if not sentence:
                continue
            await self._dispatch_sentence(sentence, pid=pid)
        if st.pending and _TRAILING_SENTENCE_END.search(st.pending):
            sentence  = st.pending.strip()
            st.pending = ""
            if sentence:
                await self._dispatch_sentence(sentence, pid=pid)

    async def _dispatch_sentence(self, sentence: str, *, pid: str) -> None:
        logger.info("tts sentence dispatch pid={!r} len={}", pid, len(sentence))
        queue = self._ensure_sender(pid)
        st = self._state(pid)
        st.synth_seq += 1
        task  = asyncio.create_task(
            self._tts.synthesize(sentence),
            name=f"tts-synth-{pid}-{st.synth_seq}",
        )
        await queue.put((task, pid))

    async def _sender_loop(self, queue: asyncio.Queue) -> None:
        """Await each synth task in FIFO order, observe the WAV, and push the
        decoded audio downstream as ``OutputAudioRawFrame``s."""
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                task, pid = item
                try:
                    wav = await task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("tts synth failed pid={!r}", pid)
                    continue
                if not wav:
                    continue
                # Let the gate's lazy chime build pick up the sample rate from
                # real TTS output exactly once.
                try:
                    self._voice_gate.observe_tts_wav(wav)
                except Exception:
                    logger.exception("observe_tts_wav raised pid={!r}", pid)
                await self._push_wav(wav, pid=pid)
        except asyncio.CancelledError:
            return

    async def _push_wav(self, wav_bytes: bytes, *, pid: str) -> None:
        try:
            frames = wav_to_output_frames(wav_bytes, pid)
        except Exception:
            logger.exception("tts WAV decode failed pid={!r}", pid)
            return
        for out in frames:
            await self.push_frame(out)

    async def _teardown_sender(self, st: _TtsPidState) -> None:
        """Cancel one participant's sender task and drop any parked synth tasks."""
        task = st.sender_task
        st.sender_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # expected — we cancelled it ourselves
        queue = st.sender_queue
        st.sender_queue = None
        if queue is not None:
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    continue
                synth_task, _ = item
                synth_task.cancel()

    async def _flush_hub(self, pid: str) -> None:
        """Drop already-paced hub audio for ``pid`` so a stop is immediate.

        Cancelling synth/sender tasks stops *new* audio, but anything already
        queued downstream (the hub's pacing pipe, the LiveKit jitter buffer)
        keeps playing; flushing the hub return-audio buffer drops it at source.
        """
        if self._transport is None or not pid:
            return
        logger.info("hub return-audio flushed pid={!r}", pid)
        try:
            await self._transport.endpoint.flush_return_audio(pid)
        except Exception:
            logger.opt(exception=True).debug(
                "flush_return_audio failed on interrupt pid={!r}", pid,
            )

    async def _drain_on_interrupt(self, pid: str) -> None:
        """Cancel the interrupting participant's in-flight TTS and flush its hub
        audio — without touching any other participant's stream."""
        st = self._by_pid.pop(pid, None)
        if st is not None:
            st.pending = ""
            queue_len = st.sender_queue.qsize() if st.sender_queue is not None else 0
            await self._teardown_sender(st)
            logger.info("tts sender drained pid={!r} queue_len={}", pid, queue_len)
        await self._flush_hub(pid)

    async def _drain_all_on_interrupt(self) -> None:
        """Legacy no-pid fallback: drain every participant and flush the
        fallback target participant's hub audio."""
        for pid in list(self._by_pid):
            st = self._by_pid.pop(pid)
            st.pending = ""
            await self._teardown_sender(st)
        if self._transport is not None:
            await self._flush_hub(self._transport.target_participant)

    async def _shutdown_all(self) -> None:
        """Cancel every participant's sender task on pipeline EndFrame/CancelFrame."""
        for pid in list(self._by_pid):
            st = self._by_pid.pop(pid)
            await self._teardown_sender(st)
