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
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_models import STTService
from xr_ai_vad import VadDetector


@dataclass(frozen=True)
class VadConfig:
    """Tuning knobs for the Silero-VAD utterance detector.

    Mirrors the constructor of :class:`xr_ai_vad.VadDetector`. Default
    values match the in-tree samples' current behavior.
    """
    silence_duration: float = 0.8
    min_speech:       float = 0.15
    silero_threshold: float = 0.5


class VadSttProcessor(FrameProcessor):
    """Consumes ``InputAudioRawFrame``; emits
    ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` /
    ``TranscriptionFrame``.

    A single shared ``VadDetector`` is held per-participant. The pid is
    read from ``frame.transport_source`` (pipecat's standard hook for
    "which input track did this come from"); an unset transport_source
    is treated as the single anonymous participant ``""`` — adequate for
    samples that only ever expect one speaker at a time and matches the
    transport's current single-target behavior.
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

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    def _detector_for(self, pid: str) -> VadDetector:
        det = self._detectors.get(pid)
        if det is not None:
            return det

        async def on_speech_start() -> None:
            self._current_pid = pid
            await self.push_frame(UserStartedSpeakingFrame())

        async def on_utterance(audio_bytes: bytes, sample_rate: int) -> None:
            # Order matters: pipecat consumers expect "user stopped speaking"
            # before the transcript so they can finalize turn state.
            await self.push_frame(UserStoppedSpeakingFrame())
            try:
                text = await self._stt.transcribe(audio_bytes, sample_rate=sample_rate)
            except Exception:
                logger.exception("stt transcribe failed pid={!r}", pid)
                return
            if not text:
                return
            await self.push_frame(TranscriptionFrame(
                text      = text,
                user_id   = pid,
                timestamp = _now_iso(),
            ))

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
        pid = frame.transport_source or ""
        det = self._detector_for(pid)
        await det.feed(frame.audio, frame.sample_rate)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
