# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VadDetector — async Silero VAD utterance accumulator.

Canonical input is raw int16 LE PCM bytes at 16 kHz (mono).  The
``on_utterance`` callback fires with the same int16 PCM bytes when speech
ends; an optional ``on_speech_start`` fires the moment cumulative speech
crosses ``min_speech`` for speculative downstream warmup.

Usage::

    async def handle_utterance(audio_bytes: bytes, sample_rate: int) -> None:
        # audio_bytes is int16 PCM, ready for WAV / STT.
        ...

    vad = VadDetector(on_utterance=handle_utterance, silero_threshold=0.5)
    await vad.feed(int16_pcm_bytes, sample_rate=16_000)
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

import numpy as np
import torch
from silero_vad import load_silero_vad

log = logging.getLogger("xr_ai_vad")

# Silero v5 only accepts 16 kHz or 8 kHz. We always classify at 16 kHz —
# complete original-rate windows are linearly resampled for classification.
# The original-rate PCM still goes to `on_utterance` so STT /
# WAV downstream get full-fidelity audio.
_SILERO_SR       = 16_000
_SILERO_WINDOW   = 512    # 32 ms at 16 kHz — silero v5 hard requirement
_MAX_UTT_S       = 30.0
_PRE_ROLL_WINDOWS = 10    # 320 ms pre-roll (10 × 32 ms)


OnUtteranceCb   = Callable[[bytes, int], Awaitable[None]]
"""Async callback receiving completed int16 PCM bytes and their sample rate."""

OnSpeechStartCb = Callable[[], Awaitable[None]]
"""Async callback fired once when an utterance crosses the speech threshold."""


class VadDetector:
    """Per-participant Silero VAD + utterance accumulator.

    ``feed()`` accepts raw int16 LE PCM bytes.  When a complete utterance is
    detected (silence after ``min_speech``) the ``on_utterance`` callback is
    awaited with int16 PCM bytes and the sample rate.

    When ``on_speech_start`` is provided, it fires once per utterance at the
    moment ``speech_s`` first crosses ``min_speech`` — a "leading edge" hook
    for speculative work (e.g. warming up a downstream resource before STT
    completes). It is awaited on the feeding task before processing further
    windows or finalizing that utterance. Callback exceptions are logged and
    isolated so detection can continue.
    """

    def __init__(
        self,
        on_utterance:      OnUtteranceCb,
        *,
        on_speech_start:   Optional[OnSpeechStartCb] = None,
        silence_duration:  float = 0.8,
        min_speech:        float = 0.15,
        silero_threshold:  float = 0.5,
    ) -> None:
        self._on_utterance      = on_utterance
        self._on_speech_start   = on_speech_start
        self._silence_duration  = silence_duration
        self._min_speech        = min_speech
        self._silero_threshold  = silero_threshold

        self._buffer:         list[bytes] = []
        self._buffer_samples: int         = 0
        self._speech_s:       float       = 0.0
        self._silent_s:       float       = 0.0
        self._speaking:       bool        = False
        # One-shot per utterance: ensures on_speech_start fires at most
        # once between finalizations.
        self._speech_start_fired: bool = False

        # Rolling pre-roll contains complete classification windows.
        self._pre_roll: list[bytes] = []

        # Silero state.  onnx=True keeps the wheel light; we still need torch
        # because silero's __call__ expects a torch.Tensor.
        self._silero     = load_silero_vad(onnx=True)
        self._pending_pcm = bytearray()
        self._window_phase = 0
        log.info("Silero VAD loaded (onnx=True)")

    def reset(self) -> None:
        """Drop any in-progress utterance without emitting it."""
        self._buffer.clear()
        self._buffer_samples     = 0
        self._speech_s           = 0.0
        self._silent_s           = 0.0
        self._speaking           = False
        self._speech_start_fired = False
        self._pre_roll.clear()
        self._pending_pcm.clear()
        self._window_phase = 0
        self._silero.reset_states()

    async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
        """Process one chunk of int16 LE PCM audio.

        ``pcm_int16``   — raw int16 little-endian PCM bytes (mono).
        ``sample_rate`` — sample rate in Hz (constant for this stream).

        Incomplete classification windows remain buffered. Speech and silence
        durations advance only when a complete 32 ms window is classified,
        independently of the caller's chunk size.
        """
        self._pending_pcm.extend(pcm_int16)
        while True:
            # Carry fractional source samples so rates such as 44.1 kHz do
            # not drift relative to Silero's 512-sample classification clock.
            n_samples, phase = divmod(
                self._window_phase + sample_rate * _SILERO_WINDOW, _SILERO_SR,
            )
            n_bytes = n_samples * 2
            if n_bytes == 0 or len(self._pending_pcm) < n_bytes:
                return
            window = bytes(self._pending_pcm[:n_bytes])
            del self._pending_pcm[:n_bytes]
            self._window_phase = phase
            await self._feed_window(window, sample_rate)

    async def _feed_window(self, pcm_int16: bytes, sample_rate: int) -> None:
        n_samples = len(pcm_int16) // 2
        chunk_s = n_samples / sample_rate

        is_speech = self._classify(pcm_int16, sample_rate)

        if is_speech:
            if not self._speaking:
                log.debug("speech START")
                self._speaking = True
                if self._pre_roll:
                    pre = b"".join(self._pre_roll)
                    self._buffer.insert(0, pre)
                    self._buffer_samples += len(pre) // 2
                    self._pre_roll.clear()
            self._buffer.append(pcm_int16)
            self._buffer_samples += n_samples
            prev_speech_s = self._speech_s
            self._speech_s += chunk_s
            self._silent_s  = 0.0

            if (self._on_speech_start is not None
                    and not self._speech_start_fired
                    and self._speech_s >= self._min_speech
                    and prev_speech_s < self._min_speech):
                self._speech_start_fired = True
                await self._safe_speech_start()
        else:
            if self._speaking:
                self._buffer.append(pcm_int16)
                self._buffer_samples += n_samples
                self._silent_s += chunk_s
            else:
                self._pre_roll.append(pcm_int16)
                if len(self._pre_roll) > _PRE_ROLL_WINDOWS:
                    self._pre_roll.pop(0)

        utt_s = self._buffer_samples / max(sample_rate, 1)
        if self._speaking and utt_s >= _MAX_UTT_S:
            log.info("VAD: max utterance length reached (%.1fs) — finalizing", utt_s)
            await self._finalize(sample_rate)
            return

        if (
            self._speaking
            and self._speech_s >= self._min_speech
            and self._silent_s >= self._silence_duration
        ):
            await self._finalize(sample_rate)

    def _classify(self, pcm_int16: bytes, sample_rate: int) -> bool:
        """Classify one complete 32 ms window, resampling only for Silero."""
        f32 = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != _SILERO_SR:
            # Original-rate PCM stays intact for the utterance callback.
            f32 = np.interp(
                np.linspace(0.0, f32.size - 1, _SILERO_WINDOW, dtype=np.float32),
                np.arange(f32.size, dtype=np.float32),
                f32,
            ).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(f32))
        return float(self._silero(tensor, _SILERO_SR)) > self._silero_threshold

    async def _safe_speech_start(self) -> None:
        try:
            assert self._on_speech_start is not None
            await self._on_speech_start()
        except Exception:
            # isolation boundary: a user callback must not stop detection.
            log.exception("on_speech_start callback raised")

    async def _finalize(self, sample_rate: int) -> None:
        if not self._buffer:
            self._speaking           = False
            self._speech_start_fired = False
            return
        audio_bytes              = b"".join(self._buffer)
        self._buffer             = []
        self._buffer_samples     = 0
        self._speaking           = False
        self._silent_s           = 0.0
        self._speech_s           = 0.0
        self._speech_start_fired = False
        # Reset Silero's RNN hidden state so the next utterance's silence
        # detection isn't biased by this utterance's speech context.
        self._silero.reset_states()
        dur_s = (len(audio_bytes) // 2) / max(sample_rate, 1)
        log.info("VAD: utterance finalized  dur=%.2fs", dur_s)
        try:
            await self._on_utterance(audio_bytes, sample_rate)
        except Exception:
            log.exception("on_utterance callback raised")
