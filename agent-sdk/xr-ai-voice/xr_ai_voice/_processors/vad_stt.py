# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VadSttProcessor`` — turns mic audio into transcriptions.

Lives at the head of the voice pipeline. For each
``InputAudioRawFrame`` it feeds the per-participant ``VadDetector``;
when the detector emits an utterance the processor sends it through the
injected ``STTService`` and pushes a ``TranscriptionFrame`` downstream.

VAD start/stop edges are forwarded as pipecat's built-in
``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` so the assistant
can cancel in-flight work on the moment speech starts.

Also runs bounded early STT probes shortly after speech-start. Brief STOP
utterances interrupt without waiting for VAD finalization, while an optional
partial-transcript callback can acknowledge a wake phrase before the user
finishes the command. Normal query dispatch still uses the final transcript.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_models import STTService
from xr_ai_vad import VadDetector
from xr_ai_voicegate._phrases import STOP_RE

from .._frames import ParticipantLeftFrame


# True acknowledges, False requests another bounded probe, and None rejects
# the prefix so ordinary ambient speech does not consume all probe attempts.
PartialTranscriptHandler = Callable[[str, str], Awaitable[bool | None]]
_MAX_PARTIAL_PROBES = 3
_PARTIAL_PROBE_TAIL_S = 0.12


@dataclass(frozen=True)
class VadConfig:
    """Tuning knobs for the Silero-VAD utterance detector.

    Mirrors the constructor of :class:`xr_ai_vad.VadDetector`. Default
    values match the in-tree samples' current behavior.

    ``stop_probe_after_s`` — seconds after ``on_speech_start`` to run an
    extra STT pass on the partial audio buffer. This gives STOP commands a
    fast interrupt path and lets a configured wake phrase be acknowledged
    before the utterance ends. Set to ``0`` or negative to disable probes.
    """
    silence_duration:   float = 0.8
    min_speech:         float = 0.15
    silero_threshold:   float = 0.5
    stop_probe_after_s: float = 0.25


class VadSttProcessor(FrameProcessor):
    """Consumes ``InputAudioRawFrame``; emits
    ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` /
    ``TranscriptionFrame``.

    A single shared ``VadDetector`` is held per-participant. The pid is
    read from ``frame.transport_source`` (pipecat's standard hook for
    "which input track did this come from"). An unset transport_source
    means the transport adapter regressed — there is no usable pid to
    route assistant output / return-data / return-audio back to, so the
    frame is logged and dropped rather than silently dispatched with
    ``pid=''`` (which the hub drops on the floor anyway).
    """

    def __init__(
        self,
        *,
        stt: STTService,
        vad_cfg: VadConfig,
        on_partial_transcript: PartialTranscriptHandler | None = None,
    ) -> None:
        super().__init__()
        self._stt                   = stt
        self._vad_cfg               = vad_cfg
        self._on_partial_transcript = on_partial_transcript
        self._detectors: dict[str, VadDetector] = {}
        # Track which pid is currently in an utterance so on_utterance
        # can push the matching ``UserStoppedSpeakingFrame`` even though
        # the VAD callback itself is pid-agnostic.
        self._current_pid: str | None = None
        # Per-pid mutable audio buffer for the early STOP probe — present
        # only while an utterance is in flight. ``on_speech_start`` opens
        # the entry; ``on_utterance`` (and the probe itself, after firing)
        # close it.
        self._probe_buffer:   dict[str, bytearray] = {}
        self._probe_sr:       dict[str, int]       = {}
        # One probe task per pid, so a fresh speech_start can cancel a
        # lingering task before scheduling the next.
        self._probe_task:     dict[str, asyncio.Task] = {}
        self._probe_inflight: set[str] = set()
        # Pids whose probe has already pushed a STOP for the current
        # utterance — suppresses the duplicate that would fire when VAD
        # eventually finalizes the same speech run. Cleared on the next
        # ``on_speech_start`` for the pid.
        self._stop_fired_for_current_utterance: set[str] = set()
        # Presentation timestamp (ns) of the most recent audio frame per pid, and
        # the one captured at speech onset. The onset value anchors the resulting
        # transcript to when the participant started speaking rather than to when
        # STT finished, which can be seconds later.
        self._last_frame_pts: dict[str, int] = {}
        self._utterance_pts:  dict[str, int] = {}

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            # Pipeline shutdown: tear down every pending probe task so a probe
            # cannot push a transcript (and trigger a turn) after the session has
            # ended. The frame is still forwarded so the rest of the pipeline
            # stops normally.
            for pid in list(self._probe_task):
                await self._cancel_probe_task(pid)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            # Evict all per-pid state for the departing participant so the
            # detector / buffer / flag dicts don't grow without bound over a
            # long-lived session of joins and leaves. The frame is still
            # forwarded downstream so the gate / assistant / transport can react.
            await self._evict_participant(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    def _detector_for(self, pid: str) -> VadDetector:
        det = self._detectors.get(pid)
        if det is not None:
            return det

        async def on_speech_start() -> None:
            self._current_pid = pid
            # Await the cancelled task before scheduling the next probe so
            # the two tasks never overlap mid-push_frame.
            self._stop_fired_for_current_utterance.discard(pid)
            onset_pts = self._last_frame_pts.get(pid)
            if onset_pts is not None:
                self._utterance_pts[pid] = onset_pts
            await self._cancel_probe_task(pid)
            detector = self._detectors.get(pid)
            try:
                audio_snapshot, sample_rate = (
                    detector.snapshot_utterance() if detector else (b"", 0)
                )
            except AttributeError:
                # Lightweight detector doubles may only implement ``feed``.
                audio_snapshot, sample_rate = b"", 0
            self._probe_buffer[pid] = bytearray(audio_snapshot)
            self._probe_sr[pid] = sample_rate
            if self._vad_cfg.stop_probe_after_s > 0:
                self._probe_task[pid] = asyncio.create_task(
                    self._run_partial_probes(pid),
                    name=f"vad-partial-probe-{pid}",
                )
                logger.info(
                    "early transcript probe scheduled pid={!r} after={:.2f}s",
                    pid, self._vad_cfg.stop_probe_after_s,
                )
            logger.info("speech start pid={!r}", pid)
            f = UserStartedSpeakingFrame()
            f.transport_source = pid
            await self.push_frame(f)

        async def on_utterance(audio_bytes: bytes, sample_rate: int) -> None:
            self._probe_buffer.pop(pid, None)
            self._probe_sr.pop(pid, None)
            await self._finish_probe_for_utterance(pid)

            dur_s = (len(audio_bytes) // 2) / max(sample_rate, 1)
            logger.info("utterance finalize pid={!r} dur={:.2f}s", pid, dur_s)

            # If the probe already pushed STOP for this utterance, the
            # frames downstream (InterruptionFrame + TranscriptionFrame
            # to the gate + UserStoppedSpeakingFrame) have already done
            # their job. Re-firing UserStoppedSpeakingFrame + a fresh
            # TranscriptionFrame (gate sees STOP again → re-fires the ack
            # TTS) would double the stop-ack. Suppress.
            if pid in self._stop_fired_for_current_utterance:
                self._stop_fired_for_current_utterance.discard(pid)
                logger.info(
                    "suppressed duplicate utterance after probe STOP pid={!r}", pid,
                )
                logger.debug(
                    "VadSttProcessor suppressing duplicate VAD-finalize "
                    "STOP pid={!r} (probe already fired)", pid,
                )
                return

            # Order matters: pipecat consumers expect "user stopped speaking"
            # before the transcript so they can finalize turn state.
            f = UserStoppedSpeakingFrame()
            f.transport_source = pid
            await self.push_frame(f)
            try:
                text = await self._stt.transcribe(audio_bytes, sample_rate=sample_rate)
            except Exception:
                logger.exception("stt transcribe failed pid={!r}", pid)
                return
            if not text:
                return
            tf = TranscriptionFrame(
                text      = text,
                user_id   = pid,
                timestamp = _now_iso(),
            )
            # Propagate the pid on transport_source too — downstream
            # processors that key on the pipecat-standard field (rather
            # than user_id) need the same value.
            tf.transport_source = pid
            # Anchor the transcript to speech onset, not to STT completion.
            tf.pts = self._utterance_pts.pop(pid, None)
            await self.push_frame(tf)

        det = VadDetector(
            on_utterance      = on_utterance,
            on_speech_start   = on_speech_start,
            silence_duration  = self._vad_cfg.silence_duration,
            min_speech        = self._vad_cfg.min_speech,
            silero_threshold  = self._vad_cfg.silero_threshold,
        )
        self._detectors[pid] = det
        return det

    async def _handle_audio(self, frame: InputAudioRawFrame) -> None:
        pid = frame.transport_source
        if not pid:
            # The transport adapter is responsible for populating
            # transport_source with the participant id. If it is missing
            # there is no usable routing target for any downstream
            # response — log loudly and drop rather than dispatch with
            # pid='' (which the hub would drop silently anyway).
            logger.error(
                "VadSttProcessor dropped InputAudioRawFrame with no "
                "transport_source — transport adapter regression?",
            )
            return
        det = self._detector_for(pid)

        # Recorded before ``feed`` so a synchronous ``on_speech_start`` sees the
        # pts of the chunk that triggered it.
        if frame.pts is not None:
            self._last_frame_pts[pid] = frame.pts

        await det.feed(frame.audio, frame.sample_rate)

        # Accumulate audio for the probe only while speech is active —
        # the dict entry is opened in on_speech_start and closed in
        # on_utterance (or by the probe itself after firing STOP). Append
        # AFTER ``feed`` so the chunk that synchronously triggered
        # on_speech_start lands in the buffer. (In production
        # on_speech_start runs as a task and may not have created the
        # entry yet — at most we lose ~20-30ms of audio, which is fine
        # for STOP detection on the remainder.)
        buf = self._probe_buffer.get(pid)
        if buf is not None:
            buf.extend(frame.audio)
            self._probe_sr[pid] = frame.sample_rate

    async def _run_partial_probes(self, pid: str) -> None:
        """Probe partial audio for STOP and an optional wake acknowledgement."""
        attempts = _MAX_PARTIAL_PROBES if self._on_partial_transcript else 1
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                due = started + attempt * self._vad_cfg.stop_probe_after_s
                await asyncio.sleep(max(0.0, due - time.monotonic()))
            except asyncio.CancelledError:
                return

            buf = self._probe_buffer.get(pid)
            sr = self._probe_sr.get(pid, 0)
            if not buf or sr <= 0:
                return

            audio = bytes(buf) + bytes(round(sr * _PARTIAL_PROBE_TAIL_S) * 2)
            before = time.monotonic()
            self._probe_inflight.add(pid)
            try:
                try:
                    text = await self._stt.transcribe(audio, sample_rate=sr)
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("partial-probe stt transcribe failed pid={!r}", pid)
                    return

                stop_matched = bool(text and STOP_RE.match(text))
                logger.info(
                    "early transcript probe fired pid={!r} attempt={} latency_ms={} "
                    "stop_matched={} text={!r}",
                    pid, attempt, round((time.monotonic() - before) * 1000),
                    stop_matched, text,
                )
                if stop_matched:
                    await self._emit_early_stop(pid, text)
                    return

                if text and self._on_partial_transcript is not None:
                    try:
                        decision = await self._on_partial_transcript(pid, text)
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        logger.exception("partial-transcript handler failed pid={!r}", pid)
                        return
                    if decision is True:
                        logger.info("early wake phrase acknowledged pid={!r}", pid)
                        return
                    if decision is None:
                        return
            finally:
                self._probe_inflight.discard(pid)
            if pid not in self._probe_buffer:
                return

    async def _emit_early_stop(self, pid: str, text: str) -> None:
        """Emit the interrupt sequence for a STOP matched by a partial probe."""

        # Race guard: if on_utterance already closed the buffer between
        # the STT await returning and this check, the cancellation simply
        # hasn't propagated yet. Bow out — the finalize path will handle
        # the rest.
        if pid not in self._probe_buffer:
            return

        logger.debug(
            "VadSttProcessor early-probe STOP match pid={!r}",
            pid,
        )

        # Mark before pushing so the suppression flag is set if the VAD
        # racing-finalize lands while frames are still queueing downstream.
        self._stop_fired_for_current_utterance.add(pid)

        # Close the probe buffer now — on_utterance will see the empty
        # entry and skip its own buffering work, but the suppression flag
        # is what actually gates the duplicate frame emission.
        self._probe_buffer.pop(pid, None)
        self._probe_sr.pop(pid, None)

        # Frame order intentionally differs from ``on_utterance``'s
        # USSF-first convention: the probe is a fast-path interruption,
        # not a clean end-of-turn. InterruptionFrame goes first so the
        # assistant cancels any in-flight reasoning before the gate sees the
        # STOP transcript and re-issues its own InterruptionFrame +
        # canned ack. UserStoppedSpeakingFrame tails as a hint to
        # downstream turn-state consumers that the partial-audio turn
        # has ended.
        f = InterruptionFrame()
        f.transport_source = pid
        await self.push_frame(f)
        tf = TranscriptionFrame(
            text      = text,
            user_id   = pid,
            timestamp = _now_iso(),
        )
        tf.transport_source = pid
        # Same onset anchor as the final-utterance path. Kept (not popped) so a
        # subsequent VAD finalize for the same run still carries it.
        tf.pts = self._utterance_pts.get(pid)
        await self.push_frame(tf)
        ssf = UserStoppedSpeakingFrame()
        ssf.transport_source = pid
        await self.push_frame(ssf)

    async def _evict_participant(self, pid: str) -> None:
        """Drop all per-pid state when a participant leaves.

        ``_detectors`` and the probe-related dicts/sets are keyed by pid and
        are otherwise only ever added to (on first audio / speech-start), so
        without an eviction path they grow unbounded across a session's
        join/leave churn. Cancel any live probe task first so it can't fire
        against torn-down state, then pop every per-pid entry.
        """
        await self._cancel_probe_task(pid)
        self._detectors.pop(pid, None)
        self._probe_buffer.pop(pid, None)
        self._probe_sr.pop(pid, None)
        self._stop_fired_for_current_utterance.discard(pid)
        self._last_frame_pts.pop(pid, None)
        self._utterance_pts.pop(pid, None)
        if self._current_pid == pid:
            self._current_pid = None
        logger.info("evicted per-participant VAD state pid={!r}", pid)

    async def _finish_probe_for_utterance(self, pid: str) -> None:
        """Let an active STT probe finish; cancel a probe still waiting to run."""
        task = self._probe_task.get(pid)
        if task is None or task.done() or pid not in self._probe_inflight:
            await self._cancel_probe_task(pid)
            return
        self._probe_task.pop(pid, None)
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("partial probe completion raised pid={!r}", pid)

    async def _cancel_probe_task(self, pid: str) -> None:
        """Cancel a pending probe task and await its teardown.

        Awaiting is what closes the race that produces intermittent
        ``coroutine '...__process_frame_task_handler' was never awaited``
        warnings: a cancelled probe may still be mid
        ``await self.push_frame(...)`` (frame already queued at the
        downstream processor's ``__input_queue``) when the next
        ``on_speech_start`` fires. If we don't wait for that
        cancellation to land, two probe tasks briefly overlap and
        downstream cancel-and-recreate-process-task cycles can race
        against the in-flight push, leaving the freshly-created
        downstream process-task coroutine un-scheduled.

        Swallow ``CancelledError`` from the task — the cancellation is
        ours; surfacing it would propagate back into ``on_speech_start``
        / ``on_utterance`` and abort the rest of those callbacks for no
        reason.
        """
        task = self._probe_task.pop(pid, None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Expected: we cancelled this probe ourselves; awaiting it raises
            # CancelledError, which we swallow so a fresh probe can't overlap the
            # one being torn down.
            pass
        except Exception:
            logger.exception("stop-probe cancel raised pid={!r}", pid)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
