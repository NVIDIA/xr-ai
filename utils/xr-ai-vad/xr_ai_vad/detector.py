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

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import numpy as np
import torch
from silero_vad import load_silero_vad

log = logging.getLogger("xr_ai_vad")

# Silero v5 only accepts 16 kHz or 8 kHz. We always classify at 16 kHz —
# input at any other rate is linearly resampled in-process before
# buffering. The original-rate PCM still goes to `on_utterance` so STT /
# WAV downstream get full-fidelity audio.
_SILERO_SR       = 16_000
_SILERO_WINDOW   = 512    # 32 ms at 16 kHz — silero v5 hard requirement
_MAX_UTT_S       = 30.0
_PRE_ROLL_CHUNKS = 10     # ~320 ms pre-roll (10 × 32 ms)


OnUtteranceCb   = Callable[[bytes, int], Awaitable[None]]
OnSpeechStartCb = Callable[[], Awaitable[None]]


class VadDetector:
    """Per-participant Silero VAD + utterance accumulator.

    ``feed()`` accepts raw int16 LE PCM bytes.  When a complete utterance is
    detected (silence after ``min_speech``) the ``on_utterance`` callback is
    awaited with int16 PCM bytes and the sample rate.

    When ``on_speech_start`` is provided, it fires once per utterance at the
    moment ``speech_s`` first crosses ``min_speech`` — a "leading edge" hook
    for speculative work (e.g. warming up a downstream resource before STT
    completes).
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
        self._busy:           bool        = False   # on_utterance in flight
        # One-shot per utterance: ensures on_speech_start fires at most
        # once between finalizations.
        self._speech_start_fired: bool = False

        # Rolling pre-roll: keep last N raw int16 chunks before speech onset.
        self._pre_roll: list[bytes] = []

        # Silero state.  onnx=True keeps the wheel light; we still need torch
        # because silero's __call__ expects a torch.Tensor.
        self._silero     = load_silero_vad(onnx=True)
        self._silero_buf = np.zeros(0, np.float32)
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
        self._silero_buf = np.zeros(0, np.float32)

    async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
        """Process one chunk of int16 LE PCM audio.

        ``pcm_int16``   — raw int16 little-endian PCM bytes (mono).
        ``sample_rate`` — sample rate in Hz.
        """
        n_samples = len(pcm_int16) // 2
        if n_samples == 0:
            return
        chunk_s = n_samples / max(sample_rate, 1)

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
                asyncio.create_task(self._safe_speech_start())
        else:
            if self._speaking:
                self._buffer.append(pcm_int16)
                self._buffer_samples += n_samples
                self._silent_s += chunk_s
            else:
                self._pre_roll.append(pcm_int16)
                if len(self._pre_roll) > _PRE_ROLL_CHUNKS:
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
            and not self._busy
        ):
            await self._finalize(sample_rate)

    def _classify(self, pcm_int16: bytes, sample_rate: int) -> bool:
        """Return True if the chunk is speech.

        Silero v5 only accepts 16 kHz or 8 kHz input and requires a fixed
        window size (512 samples at 16 kHz). Callers commonly hand us
        48 kHz audio straight from a WebRTC track, so we resample to
        16 kHz here and always call silero with sample_rate=16000.
        """
        f32 = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != _SILERO_SR and f32.size:
            # Linear resample is adequate for VAD — we only need rough
            # spectral profile, not audiophile fidelity. Avoids pulling in
            # scipy just for this single hop.
            n_out = max(1, int(round(f32.size * _SILERO_SR / sample_rate)))
            f32 = np.interp(
                np.linspace(0.0, f32.size - 1, n_out, dtype=np.float32),
                np.arange(f32.size, dtype=np.float32),
                f32,
            ).astype(np.float32)
        self._silero_buf = np.concatenate([self._silero_buf, f32])
        speech_prob = 0.0
        while len(self._silero_buf) >= _SILERO_WINDOW:
            window = self._silero_buf[:_SILERO_WINDOW]
            self._silero_buf = self._silero_buf[_SILERO_WINDOW:]
            tensor = torch.from_numpy(np.ascontiguousarray(window))
            speech_prob = max(speech_prob, float(self._silero(tensor, _SILERO_SR)))
        return speech_prob > self._silero_threshold

    async def _safe_speech_start(self) -> None:
        try:
            assert self._on_speech_start is not None
            await self._on_speech_start()
        except Exception:
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
        self._busy               = True
        dur_s = (len(audio_bytes) // 2) / max(sample_rate, 1)
        log.info("VAD: utterance finalized  dur=%.2fs", dur_s)
        try:
            await self._on_utterance(audio_bytes, sample_rate)
        except Exception:
            log.exception("on_utterance callback raised")
        finally:
            self._busy = False
