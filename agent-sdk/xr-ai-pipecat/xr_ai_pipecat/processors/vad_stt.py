# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VadSttProcessor`` — turns mic audio into transcriptions.

Lives at the head of the voice pipeline. For each
``InputAudioRawFrame`` it feeds the per-participant ``VadDetector``;
when the detector emits an utterance the processor sends it through the
injected ``STTService`` and pushes a ``TranscriptionFrame`` downstream.

VAD start/stop edges are forwarded as pipecat's built-in
``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` so the brain
can cancel in-flight work on the moment speech starts.

Also runs an early STT *probe* shortly after speech-start so brief
STOP utterances ("stop", "be quiet") interrupt the agent without
waiting for VAD's full silence-window finalize. The probe transcribes
the partial audio buffer; on a STOP-pattern match it pushes an
``InterruptionFrame`` immediately and lets the gate handle the canned
ack. Anything else is discarded and normal VAD-finalize handles the
query.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
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

from ..frames import ParticipantLeftFrame


@dataclass(frozen=True)
class VadConfig:
    """Tuning knobs for the Silero-VAD utterance detector.

    Mirrors the constructor of :class:`xr_ai_vad.VadDetector`. Default
    values match the in-tree samples' current behavior.

    ``stop_probe_after_s`` — seconds after ``on_speech_start`` to run an
    extra STT pass on the partial audio buffer and check for STOP. This
    gives the user a fast-path interrupt on brief commands ("stop",
    "be quiet") that VAD would otherwise hold open for the full
    ``silence_duration`` window. Set to ``0`` or negative to disable
    the probe and rely solely on VAD finalize.
    """
    silence_duration:   float = 0.8
    min_speech:         float = 0.15
    silero_threshold:   float = 0.5
    stop_probe_after_s: float = 0.4


class VadSttProcessor(FrameProcessor):
    """Consumes ``InputAudioRawFrame``; emits
    ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` /
    ``TranscriptionFrame``.

    A single shared ``VadDetector`` is held per-participant. The pid is
    read from ``frame.transport_source`` (pipecat's standard hook for
    "which input track did this come from"). An unset transport_source
    means the transport adapter regressed — there is no usable pid to
    route brain output / return-data / return-audio back to, so the
    frame is logged and dropped rather than silently dispatched with
    ``pid=''`` (which the hub drops on the floor anyway).
    """

    def __init__(self, *, stt: STTService, vad_cfg: VadConfig) -> None:
        super().__init__()
        self._stt        = stt
        self._vad_cfg    = vad_cfg
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
        # Pids whose probe has already pushed a STOP for the current
        # utterance — suppresses the duplicate that would fire when VAD
        # eventually finalizes the same speech run. Cleared on the next
        # ``on_speech_start`` for the pid.
        self._stop_fired_for_current_utterance: set[str] = set()

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
            return

        if isinstance(frame, ParticipantLeftFrame):
            # Evict all per-pid state for the departing participant so the
            # detector / buffer / flag dicts don't grow without bound over a
            # long-lived session of joins and leaves. The frame is still
            # forwarded downstream so the gate / brain / transport can react.
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
            await self._cancel_probe_task(pid)
            self._probe_buffer[pid] = bytearray()
            # The first feed after start_fired carries the real sample
            # rate; default to 0 so a missing entry later means "no audio".
            self._probe_sr[pid] = 0
            if self._vad_cfg.stop_probe_after_s > 0:
                self._probe_task[pid] = asyncio.create_task(
                    self._run_stop_probe(pid),
                    name=f"vad-stop-probe-{pid}",
                )
                logger.info(
                    "early STOP probe scheduled pid={!r} after={:.2f}s",
                    pid, self._vad_cfg.stop_probe_after_s,
                )
            logger.info("speech start pid={!r}", pid)
            f = UserStartedSpeakingFrame()
            f.transport_source = pid
            await self.push_frame(f)

        async def on_utterance(audio_bytes: bytes, sample_rate: int) -> None:
            # Probe ran-or-not, the utterance has finalized. Close the
            # probe buffer entry and cancel any pending probe (e.g.
            # silence_duration < stop_probe_after_s). Await the
            # cancellation so the probe task is fully torn down — see
            # the comment in ``on_speech_start`` for why this matters.
            await self._cancel_probe_task(pid)
            self._probe_buffer.pop(pid, None)
            self._probe_sr.pop(pid, None)

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

    async def _run_stop_probe(self, pid: str) -> None:
        """Wait ``stop_probe_after_s``, then transcribe whatever's in the
        per-pid buffer and check it against ``STOP_RE``.

        On match: push ``InterruptionFrame`` immediately, then
        ``TranscriptionFrame`` (so the gate emits its canned ack), then
        ``UserStoppedSpeakingFrame``. The duplicate-STOP suppression
        flag is set so the eventual VAD-finalize for this same utterance
        doesn't re-fire the ack.

        On a non-STOP transcript, or on STT failure / timeout, the probe
        is silent — VAD finalize will dispatch normally.
        """
        try:
            await asyncio.sleep(self._vad_cfg.stop_probe_after_s)
        except asyncio.CancelledError:
            return

        # Snapshot under the same control flow that owns the buffer dict —
        # no other coroutine writes to these entries between here and the
        # cancellation path, so a plain read is safe in the asyncio model.
        buf = self._probe_buffer.get(pid)
        sr  = self._probe_sr.get(pid, 0)
        if not buf or sr <= 0:
            # Utterance already finalized (and entry cleared) or no audio
            # ever accumulated — skip silently.
            return
        audio_snapshot = bytes(buf)

        try:
            text = await self._stt.transcribe(audio_snapshot, sample_rate=sr)
        except asyncio.CancelledError:
            # Cancelled mid-transcribe — the on_utterance path is taking
            # over. Don't push anything.
            return
        except Exception:
            logger.exception("stop-probe stt transcribe failed pid={!r}", pid)
            return

        matched = bool(text and STOP_RE.match(text))
        logger.info(
            "early STOP probe fired pid={!r} elapsed={:.2f}s matched={}",
            pid, self._vad_cfg.stop_probe_after_s, matched,
        )
        if not matched:
            return

        # Race guard: if on_utterance already closed the buffer between
        # the STT await returning and this check, the cancellation simply
        # hasn't propagated yet. Bow out — the finalize path will handle
        # the rest.
        if pid not in self._probe_buffer:
            return

        logger.debug(
            "VadSttProcessor early-probe STOP match pid={!r} after={:.2f}s",
            pid, self._vad_cfg.stop_probe_after_s,
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
        # brain cancels any in-flight reasoning before the gate sees the
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
        if self._current_pid == pid:
            self._current_pid = None
        logger.info("evicted per-participant VAD state pid={!r}", pid)

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
            # Expected: we cancelled this probe ourselves; awaiting it
            # raises CancelledError, which we intentionally swallow so a
            # fresh probe can't overlap the one being torn down.
            pass
        except Exception:
            logger.exception("stop-probe cancel raised pid={!r}", pid)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
