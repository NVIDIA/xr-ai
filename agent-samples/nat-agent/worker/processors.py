# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipecat pipeline + the three FrameProcessors for the nat-agent worker.

Pipeline: ``InputTransport → SttProcessor → NatProcessor → TtsProcessor → OutputTransport``

* ``SttProcessor`` runs VAD on the raw audio stream and fires the configured
  STT server periodically while speech is accumulating (warm-up). At the
  end of each utterance it publishes the final transcript on data topic
  ``stt.transcript`` and pushes a ``TranscriptionFrame`` downstream.

* ``NatProcessor`` calls into ``NatBackend.infer(transcript, pid)`` in the
  default thread-pool executor so the asyncio loop stays responsive,
  publishes the LLM reply on topic ``agent.response``, and pushes a
  ``TextFrame`` downstream so the TTS processor can speak it.

* ``TtsProcessor`` splits the reply into sentences and synthesises them in
  parallel via ``audio.stream_sentences_to_audio``. There is **no**
  ``STTMuteFrame`` / playback-tail-wait coordination — echo cancellation is
  handled at the client by the audio input device's AEC.

Pipecat's built-in ``allow_interruptions=True`` is kept for user-driven
barge-in (a separate concern from feedback loops).
"""
from __future__ import annotations

import asyncio
import json
import logging
import string
import time

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_agent import DataMessage

from audio import stream_sentences_to_audio
import config as cfg
from nat_backend import NatBackend
from services import SttClient, TtsClient
from transport import XRMediaHubTransport

log = logging.getLogger("nat_agent.processors")

_MAX_UTTERANCE_S = 30.0

_FILLER_PHRASES = frozenset({
    "mm-hmm", "mm hmm", "uh huh", "uh-huh", "uh", "um", "ah", "oh", "eh",
    "huh", "hmm", "ug", "yeah", "yep", "yup", "okay", "ok", "right",
    "sure", "thanks", "thank you", "you know", "i mean", "well",
})


def _now_us() -> int:
    return time.time_ns() // 1_000


def _int16_rms(int16_bytes: bytes) -> float:
    arr = np.frombuffer(int16_bytes, dtype=np.int16)
    if len(arr) == 0:
        return 0.0
    f32 = arr.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f32 * f32)))


# ── SttProcessor ─────────────────────────────────────────────────────────────

class SttProcessor(FrameProcessor):
    """
    VAD + streaming STT with end-of-speech finalisation.

    * VAD on raw audio (RMS energy) — ``silence_threshold`` is the gate.
    * While speech is accumulating, every ``stream_interval`` seconds run
      STT on the growing buffer to keep the model warm. Streaming results
      are NOT published to the client.
    * End-of-speech triggers a final STT pass; the full transcript is
      published on ``stt.transcript`` and a ``TranscriptionFrame`` is
      pushed downstream.

    No ``STTMuteFrame`` handling — client AEC handles the feedback loop.
    """

    def __init__(
        self,
        stt: SttClient,
        transport: XRMediaHubTransport,
        silence_threshold: float,
        silence_duration: float,
        min_speech: float,
        stream_interval: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stt = stt
        self._transport = transport

        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._min_speech = min_speech
        self._stream_interval = stream_interval

        self._buffer: list[bytes] = []
        self._buffer_samples = 0
        self._speech_s = 0.0
        self._silent_s = 0.0
        self._speaking = False

        self._prev_words: list[str] = []
        self._last_stream = time.monotonic()

        self._stt_busy = False
        self._finalizing = False
        # In-flight streaming warm-up task (if any). _finalize() awaits it
        # before issuing its own STT call so the two do not arrive at the
        # STT server concurrently — NeMo ASR's transcribe() shares freeze
        # state across calls, and concurrent calls crash the model with
        # ``ValueError: Cannot unfreeze partially without first freezing
        # the module``. The server now serialises internally too, but
        # waiting here also avoids burning a redundant transcribe.
        self._stream_task: asyncio.Task | None = None

        # Idle RMS telemetry — helps tune silence_threshold empirically.
        self._idle_rms_peak = 0.0
        self._idle_last_log = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._feed_audio(frame)
            return

        await self.push_frame(frame, direction)

    async def _feed_audio(self, frame: InputAudioRawFrame) -> None:
        rms = _int16_rms(frame.audio)
        chunk_samples = len(frame.audio) // 2
        chunk_s = chunk_samples / max(cfg.SAMPLE_RATE, 1)
        is_voice = rms >= self._silence_threshold
        now = time.monotonic()

        if not self._speaking:
            if rms > self._idle_rms_peak:
                self._idle_rms_peak = rms
            if now - self._idle_last_log >= 2.0:
                log.info(
                    "idle  peak_rms=%.4f  threshold=%.4f",
                    self._idle_rms_peak, self._silence_threshold,
                )
                self._idle_rms_peak = 0.0
                self._idle_last_log = now

        if is_voice:
            if not self._speaking:
                log.info("Voice start (rms=%.4f)", rms)
                self._buffer.clear()
                self._buffer_samples = 0
                self._speech_s = 0.0
                self._prev_words = []
            self._speaking = True
            self._buffer.append(frame.audio)
            self._buffer_samples += chunk_samples
            self._speech_s += chunk_s
            self._silent_s = 0.0
        else:
            if self._speaking:
                self._buffer.append(frame.audio)
                self._buffer_samples += chunk_samples
                self._silent_s += chunk_s

        utt_s = self._buffer_samples / max(cfg.SAMPLE_RATE, 1)

        if self._speaking and utt_s > _MAX_UTTERANCE_S:
            log.info("Max utterance (%.0fs), finalising…", utt_s)
            await self._finalize()
            return

        if (self._speaking
                and self._speech_s >= self._min_speech
                and self._silent_s >= self._silence_duration
                and not self._finalizing):
            await self._finalize()
            return

        if (self._speaking
                and self._speech_s >= self._min_speech
                and not self._stt_busy
                and not self._finalizing
                and now - self._last_stream >= self._stream_interval):
            self._stt_busy = True
            self._last_stream = now
            self._stream_task = asyncio.create_task(self._stream_stt())

    async def _stream_stt(self) -> None:
        try:
            if not self._buffer:
                return
            audio_bytes = b"".join(self._buffer)
            try:
                transcript = await self._stt.transcribe(
                    audio_bytes, cfg.SAMPLE_RATE, cfg.NUM_CHANNELS,
                )
            except Exception:
                log.exception("streaming STT request failed")
                return
            words = transcript.split()
            new_words = words[len(self._prev_words):]
            if new_words:
                self._prev_words = words
                log.info("stream warm-up  +%r", " ".join(new_words)[:80])
        finally:
            self._stt_busy = False

    async def _finalize(self) -> None:
        if not self._buffer:
            self._speaking = False
            return

        audio_bytes = b"".join(self._buffer)
        self._buffer.clear()
        self._buffer_samples = 0
        self._speaking = False
        self._silent_s = 0.0
        self._speech_s = 0.0
        self._finalizing = True

        # Estimate the wall-clock instant the user finished speaking.
        # _finalize is called at silence-detection time, which is the
        # moment of the last non-voice chunk plus ``silence_duration``
        # of accumulated silence. Subtracting that window back gets us
        # close to true end-of-speech — within a frame or two of error.
        # This timestamp is forwarded to NatBackend.infer via the
        # TranscriptionFrame.timestamp field so the LLM can anchor
        # video-mcp lookups to when the user actually spoke instead of
        # to wall-clock-at-tool-fire-time (which is 5-15 s later).
        user_asked_at_us = _now_us() - int(self._silence_duration * 1_000_000)

        dur_s = len(audio_bytes) // 2 / max(cfg.SAMPLE_RATE, 1)
        log.info("Transcribing %.1fs of audio (final)…", dur_s)

        # Wait for any in-flight streaming warm-up to finish before issuing
        # the final STT call. NeMo's transcribe() is not thread-safe and
        # the STT server we ship here serialises requests, so two
        # overlapping calls would either crash NeMo (older versions) or
        # queue and add latency — either way, awaiting first is correct.
        if self._stream_task is not None and not self._stream_task.done():
            try:
                await self._stream_task
            except (asyncio.CancelledError, Exception):
                pass
        self._stream_task = None

        try:
            try:
                text = await self._stt.transcribe(
                    audio_bytes, cfg.SAMPLE_RATE, cfg.NUM_CHANNELS,
                )
            except Exception:
                log.exception("final STT request failed")
                return

            if not text:
                log.info("STT returned empty transcript")
                return

            log.info("STT transcript: %r", text)
            await self._send_data("stt.transcript", text)

            lower = text.lower().strip().rstrip(string.punctuation).strip()
            if self._is_unusable(lower):
                log.info("Unusable transcript (filler/noise) — not forwarding to LLM")
                return

            # Smuggle the user_asked_at_us timestamp through the otherwise-
            # unused ``timestamp`` field. NatProcessor parses it back out.
            # Pipecat treats it as opaque metadata.
            await self.push_frame(
                TranscriptionFrame(
                    text=text, user_id="",
                    timestamp=str(user_asked_at_us),
                ),
                FrameDirection.DOWNSTREAM,
            )
        finally:
            self._prev_words = []
            self._finalizing = False

    async def _send_data(self, topic: str, text: str) -> None:
        pid = self._transport.target_participant
        if not pid:
            return
        try:
            await self._transport.send_return_data(DataMessage(
                participant_id=pid,
                topic=topic,
                pts_us=_now_us(),
                data=text.encode(),
            ))
        except Exception:
            log.exception("failed to publish data on topic %r", topic)

    @staticmethod
    def _is_unusable(lower: str) -> bool:
        if not lower:
            return True
        tokens = lower.split()
        if len(tokens) < 2:
            return True
        if lower in _FILLER_PHRASES:
            return True
        if all(t.rstrip(string.punctuation) in _FILLER_PHRASES for t in tokens):
            return True
        return False


# ── NatProcessor ─────────────────────────────────────────────────────────────

class NatProcessor(FrameProcessor):
    """
    Latest-only-replace queue in front of ``NatBackend.infer``.

    Behaviour
    ---------
    * Every incoming ``TranscriptionFrame`` becomes the *pending* transcript,
      overwriting any prior pending one. A single drain task processes one
      transcript at a time.
    * When inference completes, the drain task checks whether a newer
      transcript was enqueued in the meantime; if so the just-computed answer
      is **discarded** and the loop runs again with the newer text. This
      means the user's *latest* utterance always wins — earlier transcripts
      are never spoken back when a newer one exists.
    * The currently running ``nat.infer`` call is *not* hard-cancelled (the
      vLLM completion runs to the end on the worker thread) — but its result
      is simply dropped. We accept the wasted compute as the price of not
      plumbing per-request cancellation through NAT/LangChain/httpx/vLLM.

    Why
    ---
    Pre-fix, every TranscriptionFrame queued behind the previous one through
    NatBackend's ``_infer_lock``. With ~40 s LLM turns, three rapid utterances
    (e.g. echo from TTS playback + a real query) translated into a 2-minute
    serialised backlog before the user got any response — and Pipecat's idle
    timer eventually killed the pipeline mid-backlog. Latest-only replace
    bounds the queue at one in-flight + one pending, so a real user query
    never sits behind a phantom one.

    Status (``processing`` / ``idle``) is reported around the whole drain
    phase rather than per-inference; that way the client sees a single
    "thinking" state during a flurry of utterances rather than flickering.
    """

    def __init__(
        self,
        nat: NatBackend,
        transport: XRMediaHubTransport,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._nat = nat
        self._transport = transport

        # Pending = (text, pid, ref_us). Replaced wholesale on each new
        # TranscriptionFrame. Set under _lock; cleared by the drain task
        # immediately after it picks the value up.
        self._pending: tuple[str, str, int] | None = None
        self._lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._enqueue(frame)
            return

        await self.push_frame(frame, direction)

    async def _enqueue(self, frame: TranscriptionFrame) -> None:
        text = frame.text.strip()
        if not text:
            return

        pid = self._transport.target_participant

        # SttProcessor smuggles the end-of-speech wall-clock timestamp
        # through the unused ``timestamp`` string field. Fall back to
        # "now" if it's missing.
        try:
            ref_us = int(frame.timestamp) if frame.timestamp else _now_us()
        except (TypeError, ValueError):
            ref_us = _now_us()

        async with self._lock:
            superseded = self._pending is not None
            self._pending = (text, pid, ref_us)
            if superseded:
                log.info("NAT pending REPLACED — older transcript dropped, "
                         "new=%r", text[:80])
            else:
                log.info("NAT queued  %r", text[:80])

            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(
                    self._drain(), name="nat-drain",
                )

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        announced_pid: str | None = None
        try:
            while True:
                async with self._lock:
                    if self._pending is None:
                        return
                    text, pid, ref_us = self._pending
                    self._pending = None

                # Announce 'processing' once when a drain run begins; only
                # re-announce if the participant changes mid-drain.
                if pid and pid != announced_pid:
                    await self._transport.endpoint.set_status("processing", pid)
                    announced_pid = pid

                log.info(
                    "NAT inference START  pid=%r  ref=%d  transcript=%r",
                    pid, ref_us, text[:80],
                )
                try:
                    answer = await loop.run_in_executor(
                        None, self._nat.infer, text, pid, ref_us,
                    )
                except Exception:
                    log.exception("NAT inference failed  text=%r", text[:80])
                    continue

                # If a newer transcript came in while the LLM was running,
                # discard this answer — the next loop iteration will run on
                # the fresher text. The user's latest intent wins.
                async with self._lock:
                    if self._pending is not None:
                        log.info(
                            "NAT result DISCARDED — newer transcript "
                            "pending  stale_text=%r",
                            text[:80],
                        )
                        continue

                log.info("NAT response  %d chars  text=%r",
                         len(answer), answer[:200])
                if pid and answer:
                    await self._transport.send_return_data(DataMessage(
                        participant_id=pid,
                        topic="agent.response",
                        pts_us=_now_us(),
                        data=answer.encode(),
                    ))
                await self.push_frame(
                    TextFrame(text=answer), FrameDirection.DOWNSTREAM,
                )
        finally:
            if announced_pid:
                await self._transport.endpoint.set_status("idle", announced_pid)


# ── TtsProcessor ─────────────────────────────────────────────────────────────

class TtsProcessor(FrameProcessor):
    """
    On a ``TextFrame``:
      Split into sentences, synthesise in parallel, send ``AudioChunk``s in
      sentence order via ``send_return_audio``.

    No ``STTMuteFrame``, no playback-tail wait — client AEC owns feedback
    cancellation.
    """

    def __init__(
        self,
        tts: TtsClient,
        transport: XRMediaHubTransport,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tts = tts
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        text = frame.text.strip()
        if not text:
            await self.push_frame(frame, direction)
            return

        pid = self._transport.target_participant
        if not pid:
            log.warning("TTS skipped — no target participant set")
            await self.push_frame(frame, direction)
            return

        log.info("TTS streaming  pid=%r  %d chars", pid, len(text))
        try:
            await stream_sentences_to_audio(
                self._transport.endpoint, self._tts.synthesize, text, pid,
            )
        except Exception:
            log.exception("TTS streaming failed  pid=%r", pid)

        await self.push_frame(frame, direction)


# ── pipeline factory ─────────────────────────────────────────────────────────

def build_pipeline(
    transport: XRMediaHubTransport,
    stt: SttClient,
    tts: TtsClient,
    nat: NatBackend,
    cfg_obj,
) -> tuple[Pipeline, PipelineTask]:
    stt_proc = SttProcessor(
        stt, transport,
        silence_threshold=cfg_obj.silence_threshold,
        silence_duration=cfg_obj.silence_duration,
        min_speech=cfg_obj.min_speech,
        stream_interval=cfg_obj.stream_interval,
    )
    nat_proc = NatProcessor(nat, transport)
    tts_proc = TtsProcessor(tts, transport)

    pipeline = Pipeline([
        transport.input(),
        stt_proc,
        nat_proc,
        tts_proc,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
        ),
        # None disables the idle timer outright.  Setting it to 0 means
        # asyncio.wait_for(..., timeout=0) raises TimeoutError immediately and
        # the pipeline cancels every time we go quiet — which kills the agent
        # mid-inference.  In Pipecat 1.1.0 this is a PipelineTask kwarg, not a
        # PipelineParams field; PipelineParams silently ignored it before.
        idle_timeout_secs=None,
    )

    return pipeline, task
