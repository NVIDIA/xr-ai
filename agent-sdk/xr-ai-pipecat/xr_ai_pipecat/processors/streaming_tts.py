# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``StreamingTtsProcessor`` — sentence-batched parallel TTS.

Consumes ``TextFrame``s (brain output, one per chunk/token) and emits
``OutputAudioRawFrame``s. Buffers text until a sentence boundary, kicks
off TTS for each sentence in parallel, then streams the WAVs out in
order so playback is monotonic.

``InterruptionFrame`` cancels every in-flight synth task and clears the
output queue so the user does not hear stale audio after asking the
agent to stop.

Every synthesized WAV is offered to ``VoiceGate.observe_tts_wav`` so
the gate's listening chime can lazily build at the right sample rate.

When constructed with a non-empty ``text_topic`` and a ``transport``,
the processor also echoes each brain turn's full assembled response on
the data channel under that topic — the moment it sees a
:class:`BrainResponseEndFrame`. Samples whose brain already pushes its
own per-turn data echo (e.g. xr-render-demo) pass ``text_topic=""`` to
opt out of this and avoid duplicate sends.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_agent import DataMessage
from xr_ai_models import TTSService
from xr_ai_voicegate import VoiceGate

from ..audio import wav_to_chunks
from ..frames import BrainResponseEndFrame

if TYPE_CHECKING:
    from ..transport import XRMediaHubTransport


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Matches a sentence-final char optionally followed by closing punctuation
# (quote, single quote, paren, bracket) and trailing whitespace. The
# brain may emit `'... what am I looking at?"'` — the `?` is the
# sentence end but the buffer's last char is `"`, so an `endswith(.!?)`
# check would leave the tail in pending and the next turn's text would
# be concatenated onto it.
_TRAILING_SENTENCE_END = re.compile(r"""[.!?]["')\]]*\s*$""")


class StreamingTtsProcessor(FrameProcessor):
    """Sentence-batched parallel TTS at the tail of the voice pipeline.

    Lifts the parallel-synth-with-ordered-send pattern from
    :func:`xr_ai_pipecat.audio.stream_sentences_to_audio` and exposes it
    as a frame processor.

    ``transport`` and ``text_topic`` are optional; when both are
    supplied (and the topic is non-empty), the processor emits one
    ``send_return_data`` per :class:`BrainResponseEndFrame` so the
    client receives the full assembled reply on the data channel.
    Leaving them out (or passing an empty topic) disables the echo —
    used by samples whose brain already sends its own per-turn data
    response.
    """

    def __init__(
        self,
        *,
        tts: TTSService,
        voice_gate: VoiceGate,
        transport: "XRMediaHubTransport | None" = None,
        text_topic: str = "",
    ) -> None:
        super().__init__()
        self._tts        = tts
        self._voice_gate = voice_gate
        self._transport  = transport
        self._text_topic = text_topic
        # Pending text we haven't yet split into a sentence — accumulates
        # across consecutive TextFrames so e.g. token streams from an
        # LLM coalesce into whole sentences before TTS is invoked.
        self._pending: str       = ""
        self._sender_task: asyncio.Task | None = None
        self._sender_queue: asyncio.Queue | None = None
        # Monotonic per-instance counter for unique synth task names — a
        # wall-clock millisecond stamp collides when two sentences dispatch
        # within the same millisecond.
        self._synth_seq: int = 0

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._drain_on_interrupt()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BrainResponseEndFrame):
            await self._handle_response_end(frame)
            # Forward the marker so any tail processor / sink that
            # tracks turn boundaries (tests, future debug taps) still
            # sees it.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            await self._handle_text(frame)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    def _ensure_sender(self) -> asyncio.Queue:
        """Spin up the ordered-sender task lazily on first sentence."""
        if self._sender_queue is None or self._sender_task is None or self._sender_task.done():
            self._sender_queue = asyncio.Queue()
            self._sender_task  = asyncio.create_task(
                self._sender_loop(self._sender_queue),
                name="tts-sender",
            )
        return self._sender_queue

    async def _handle_text(self, frame: TextFrame) -> None:
        if not frame.text:
            return
        # A trailing space matters when token-stream text frames arrive
        # without their own punctuation — preserve whatever the producer
        # sent verbatim.
        self._pending += frame.text
        await self._flush_complete_sentences(pid=frame.transport_destination or "")

    async def _handle_response_end(self, frame: BrainResponseEndFrame) -> None:
        """Flush trailing pending text, then send the data echo.

        The brain may finish a turn with text that has no
        sentence-final punctuation (e.g. an aborted partial answer);
        the boundary regex would leave that fragment in the buffer
        forever. End-of-response is the right place to flush it so the
        user hears the tail of the reply.
        """
        if self._pending.strip():
            sentence = self._pending.strip()
            self._pending = ""
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

    async def _flush_complete_sentences(self, *, pid: str) -> None:
        """Drain every complete sentence in the pending buffer, leaving
        any trailing fragment in place until more text arrives. A buffer
        that already ends in sentence-final punctuation is flushed in
        one shot — covers the brain-returns-a-complete-string case
        where no trailing whitespace ever arrives."""
        while True:
            m = _SENTENCE_END.search(self._pending)
            if m is None:
                break
            sentence  = self._pending[: m.end()].strip()
            self._pending = self._pending[m.end() :]
            if not sentence:
                continue
            await self._dispatch_sentence(sentence, pid=pid)
        # Pending buffer ends in sentence-final punctuation? Flush it
        # too — the brain is done writing and there's no follow-up
        # whitespace to fire the boundary regex. ``_TRAILING_SENTENCE_END``
        # tolerates trailing closing quotes/brackets after the sentence
        # char (e.g. ``... what am I looking at?"``) which a plain
        # ``endswith((".", "!", "?"))`` would miss, leaving the tail in
        # pending until it concatenated onto the next turn's reply.
        if self._pending and _TRAILING_SENTENCE_END.search(self._pending):
            sentence  = self._pending.strip()
            self._pending = ""
            if sentence:
                await self._dispatch_sentence(sentence, pid=pid)

    async def _dispatch_sentence(self, sentence: str, *, pid: str) -> None:
        logger.info("tts sentence dispatch pid={!r} len={}", pid, len(sentence))
        queue = self._ensure_sender()
        self._synth_seq += 1
        task  = asyncio.create_task(
            self._tts.synthesize(sentence),
            name=f"tts-synth-{pid}-{self._synth_seq}",
        )
        await queue.put((task, pid))

    async def _sender_loop(self, queue: asyncio.Queue) -> None:
        """Awaits each synth task in FIFO order, observes the WAV, and
        pushes the decoded audio downstream as ``OutputAudioRawFrame``s."""
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
                # Let the gate's lazy chime build pick up the sample rate
                # from real TTS output exactly once.
                try:
                    self._voice_gate.observe_tts_wav(wav)
                except Exception:
                    logger.exception("observe_tts_wav raised pid={!r}", pid)
                await self._push_wav(wav, pid=pid)
        except asyncio.CancelledError:
            return

    async def _push_wav(self, wav_bytes: bytes, *, pid: str) -> None:
        try:
            chunks = wav_to_chunks(wav_bytes, pid)
        except Exception:
            logger.exception("tts WAV decode failed pid={!r}", pid)
            return
        for c in chunks:
            f32 = np.frombuffer(c.data, dtype=np.float32)
            i16 = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            out = OutputAudioRawFrame(
                audio        = i16,
                sample_rate  = c.sample_rate,
                num_channels = c.channels,
            )
            out.transport_destination = pid
            await self.push_frame(out)

    async def _drain_on_interrupt(self) -> None:
        """Cancel everything in flight + flush the pending buffer.

        Cancelling the synth + sender tasks stops *new* audio from being
        produced, but anything already queued downstream — the hub's
        pacing pipe and the LiveKit jitter buffer — keeps playing. The
        user hears the agent finish its current sentence(s) before
        silence, and STOP feels broken. Flushing the hub's return-audio
        buffer on the way out drops that pending audio at the source so
        the stop is immediate.
        """
        self._pending = ""
        # Snapshot the target pid + queue length before tearing down so
        # the user-facing log carries useful breadcrumbs (which
        # participant was being addressed, how much queued audio is
        # about to be dropped) without exposing transcript content.
        target_pid = ""
        if self._transport is not None:
            target_pid = self._transport.target_participant
        queue_len = self._sender_queue.qsize() if self._sender_queue is not None else 0
        if self._sender_task is not None and not self._sender_task.done():
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                # Expected: we cancelled this task ourselves; awaiting it
                # raises CancelledError, which we intentionally swallow so
                # interrupt drain completes cleanly.
                pass
        # Drop any tasks still parked in the queue so they don't keep
        # running and emit audio after the interrupt.
        if self._sender_queue is not None:
            while not self._sender_queue.empty():
                try:
                    item = self._sender_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    continue
                task, _ = item
                task.cancel()
        logger.info(
            "tts sender drained pid={!r} queue_len={}", target_pid, queue_len,
        )
        self._sender_queue = None
        self._sender_task  = None

        # Flush the hub's return-audio buffer so already-paced audio
        # stops immediately instead of finishing whatever sentence was
        # mid-playback. Single-participant samples set
        # ``target_participant`` on ``ParticipantJoinedFrame``; if the
        # transport isn't wired or no participant is bound yet, there's
        # nothing to flush.
        if self._transport is not None:
            pid = self._transport.target_participant
            if pid:
                logger.info("hub return-audio flushed pid={!r}", pid)
                try:
                    await self._transport.endpoint.flush_return_audio(pid)
                except Exception:
                    logger.opt(exception=True).debug(
                        "flush_return_audio failed on interrupt pid={!r}",
                        pid,
                    )
