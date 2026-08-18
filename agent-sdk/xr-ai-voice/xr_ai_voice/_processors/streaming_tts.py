# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``StreamingTtsProcessor`` — per-participant sentence-batched TTS.

Consumes ``TextFrame``s (assistant output, one per chunk/token) and emits
``OutputAudioRawFrame``s. Buffers text per participant until a sentence
boundary, then consumes streaming PCM when the TTS service supports it or
falls back to complete-WAV synthesis. Playback stays monotonic per participant.

All streaming state — pending text, the synthesis queue, and the sender task
— is keyed by participant id, so concurrent participants never share a buffer
(which would interleave their words) or a sender (which would misroute audio).

An ``InterruptionFrame`` cancels only the interrupting participant's in-flight
synthesis and flushes only their hub audio (``transport_source`` carries the
pid); a frame with no pid falls back to draining every participant and flushing
each of their hub audio. A ``ParticipantLeftFrame`` releases just that
participant's state, and a pipeline ``EndFrame``/``CancelFrame`` tears down all
sender tasks.

WAV responses and streaming PCM metadata are offered to ``VoiceGate`` so its
listening chime can lazily build at the right sample rate. When constructed
with a non-empty ``text_topic`` and a ``transport``, the processor also echoes
each assistant turn's full assembled response on the data channel under that
topic.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import nemo_relay
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_hub import DataMessage
from xr_ai_models import StreamingTTSService, TTSAudioChunk, TTSService
from xr_ai_voicegate import VoiceGate

from .._audio import wav_to_output_frames
from .._frames import AssistantResponseEndFrame, ParticipantLeftFrame

if TYPE_CHECKING:
    from .._transport import HubVoiceTransport


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A sentence-final char optionally followed by closing punctuation (quote,
# paren, bracket) and trailing whitespace, e.g. ``... looking at?"``. A plain
# ``endswith((".", "!", "?"))`` would miss the trailing quote and leave the tail
# in the buffer to concatenate onto the next turn.
_TRAILING_SENTENCE_END = re.compile(r"""[.!?]["')\]]*\s*$""")
_PAUSE_FRAME_MS = 20
_PCM_SAMPLE_WIDTH = 2


class _TtsPidState:
    """Per-participant streaming state: pending text + its ordered sender."""

    __slots__ = (
        "pending",
        "sender_task",
        "sender_queue",
        "sentence_count",
        "synth_seq",
    )

    def __init__(self) -> None:
        self.pending: str = ""
        self.sender_task: asyncio.Task | None = None
        self.sender_queue: asyncio.Queue | None = None
        self.sentence_count: int = 0
        self.synth_seq: int = 0


class StreamingTtsProcessor(FrameProcessor):
    """Per-participant sentence-batched TTS at the pipeline tail.

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
        inter_sentence_pause_ms: int = 0,
    ) -> None:
        super().__init__()
        if inter_sentence_pause_ms < 0:
            raise ValueError("inter_sentence_pause_ms must be non-negative")
        self._tts        = tts
        self._voice_gate = voice_gate
        self._transport  = transport
        self._text_topic = text_topic
        self._inter_sentence_pause_ms = inter_sentence_pause_ms
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

        if isinstance(frame, ParticipantLeftFrame):
            # Release the departing participant's synthesis state before the
            # frame reaches the output transport (which drops that pid's media
            # sender). Without this, a lingering synth task could still emit
            # audio for the departed pid afterwards, and the transport's lazy
            # routing would recreate the sender it had just released.
            await self._drain_on_interrupt(frame.participant_id, flush=False)
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
        if st is not None:
            st.sentence_count = 0

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
        pause_before = st.sentence_count > 0
        st.sentence_count += 1
        if isinstance(self._tts, StreamingTTSService):
            await queue.put(
                (
                    "pcm",
                    self._stream_synthesize(sentence, pid=pid),
                    pid,
                    pause_before,
                )
            )
            return
        st.synth_seq += 1
        task  = asyncio.create_task(
            self._synthesize(sentence, pid=pid),
            name=f"tts-synth-{pid}-{st.synth_seq}",
            context=nemo_relay.fork_asyncio_context(),
        )
        await queue.put(("wav", task, pid, pause_before))

    async def _stream_synthesize(
        self,
        text: str,
        *,
        pid: str,
    ) -> AsyncIterator[TTSAudioChunk]:
        with nemo_relay.scope.scope(
            "voice.tts",
            nemo_relay.ScopeType.Function,
            input={"text": text},
            metadata={"participant_id": pid or None},
        ):
            if not isinstance(self._tts, StreamingTTSService):
                raise TypeError("streaming TTS adapter lost streaming support")
            async for chunk in self._tts.stream_synthesize(text):
                yield chunk

    async def _synthesize(self, text: str, *, pid: str) -> bytes:
        with nemo_relay.scope.scope(
            "voice.tts",
            nemo_relay.ScopeType.Function,
            input={"text": text},
            metadata={"participant_id": pid or None},
        ):
            return await self._tts.synthesize(text)

    async def _sender_loop(self, queue: asyncio.Queue) -> None:
        """Consume each queued sentence in order and push its audio downstream."""
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                kind, source, pid, pause_before = item
                if kind == "pcm":
                    await self._push_pcm_stream(
                        source,
                        pid=pid,
                        pause_before=pause_before,
                    )
                    continue
                task = source
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
                await self._push_wav(
                    wav,
                    pid=pid,
                    pause_before=pause_before,
                )
        except asyncio.CancelledError:
            return

    async def _push_pcm_stream(
        self,
        stream: AsyncIterator[TTSAudioChunk],
        *,
        pid: str,
        pause_before: bool,
    ) -> None:
        pause_pending = pause_before
        try:
            async for chunk in stream:
                if not chunk.data:
                    continue
                if pause_pending:
                    await self._push_silence(
                        sample_rate=chunk.sample_rate,
                        channels=chunk.channels,
                        pid=pid,
                    )
                    pause_pending = False
                await self._push_pcm(chunk, pid=pid)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("tts stream failed pid={!r}", pid)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("tts stream close failed pid={!r}", pid)

    async def _push_pcm(self, chunk: TTSAudioChunk, *, pid: str) -> None:
        if not chunk.data:
            return
        self._voice_gate.observe_tts_sample_rate(chunk.sample_rate)
        out = OutputAudioRawFrame(
            audio=chunk.data,
            sample_rate=chunk.sample_rate,
            num_channels=chunk.channels,
        )
        out.transport_destination = pid
        await self.push_frame(out)

    async def _push_wav(
        self,
        wav_bytes: bytes,
        *,
        pid: str,
        pause_before: bool,
    ) -> None:
        try:
            frames = wav_to_output_frames(wav_bytes, pid)
        except Exception:
            logger.exception("tts WAV decode failed pid={!r}", pid)
            return
        if pause_before and frames:
            await self._push_silence(
                sample_rate=frames[0].sample_rate,
                channels=frames[0].num_channels,
                pid=pid,
            )
        for out in frames:
            await self.push_frame(out)

    async def _push_silence(
        self,
        *,
        sample_rate: int,
        channels: int,
        pid: str,
    ) -> None:
        remaining = sample_rate * self._inter_sentence_pause_ms // 1000
        frame_samples = max(1, sample_rate * _PAUSE_FRAME_MS // 1000)
        while remaining > 0:
            samples = min(frame_samples, remaining)
            out = OutputAudioRawFrame(
                audio=b"\x00" * (samples * channels * _PCM_SAMPLE_WIDTH),
                sample_rate=sample_rate,
                num_channels=channels,
            )
            out.transport_destination = pid
            await self.push_frame(out)
            remaining -= samples

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
                kind, source, _, _ = item
                if kind == "wav":
                    source.cancel()
                else:
                    close = getattr(source, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except Exception:
                            logger.exception("queued tts stream close failed")

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

    async def _drain_on_interrupt(self, pid: str, *, flush: bool = True) -> None:
        """Cancel one participant's in-flight TTS without touching any other's.

        ``flush`` also drops that participant's already-paced hub audio — right
        for an interruption, pointless for a participant who has already left.
        """
        st = self._by_pid.pop(pid, None)
        if st is not None:
            st.pending = ""
            queue_len = st.sender_queue.qsize() if st.sender_queue is not None else 0
            await self._teardown_sender(st)
            logger.info("tts sender drained pid={!r} queue_len={}", pid, queue_len)
        if flush:
            await self._flush_hub(pid)

    async def _drain_all_on_interrupt(self) -> None:
        """No-pid fallback: drain every participant's synthesis and flush the
        hub audio of each of them.

        Flushing only the fallback ``target_participant`` would leave audio that
        is already paced for the other active participants playing out after a
        global interruption.
        """
        pids = list(self._by_pid)
        for pid in pids:
            st = self._by_pid.pop(pid)
            st.pending = ""
            await self._teardown_sender(st)
        if self._transport is None:
            return
        # Include the fallback target: audio pushed with no destination is paced
        # under that pid and has no entry of its own in ``_by_pid``.
        targets = list(dict.fromkeys([*pids, self._transport.target_participant]))
        for pid in targets:
            await self._flush_hub(pid)

    async def _shutdown_all(self) -> None:
        """Cancel every participant's sender task on pipeline EndFrame/CancelFrame."""
        for pid in list(self._by_pid):
            st = self._by_pid.pop(pid)
            await self._teardown_sender(st)
