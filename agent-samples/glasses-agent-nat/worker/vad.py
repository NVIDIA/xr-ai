# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VadDetector — standalone async Silero VAD (with adaptive energy fallback).

Ported from xr-ai-render-test/agent-samples/xr-render-demo/worker/processors.py
(SttProcessor) and adapted to work without Pipecat:

- Silero VAD (onnx=True) with fallback to adaptive energy-based detection.
- 320 ms pre-roll buffer (10 chunks of ~32 ms each).
- Configurable min_speech, silence_duration, silence_threshold, silero_threshold.
- _is_filler() filters common non-speech utterances.
- Async callback: on_utterance(audio_bytes, sample_rate) fires when a full
  utterance is finalized.  audio_bytes is **int16 PCM** (suitable for WAV / STT).
- Hard cap: max 30 s per utterance.

Audio format
------------
The XR hub sends AudioChunk.data as float32 LE, interleaved.  feed() accepts
that raw float32 bytes format and internally converts to int16 when building
the utterance buffer (matching the Pipecat SttProcessor's int16 path).

Usage::

    async def handle_utterance(audio_bytes: bytes, sample_rate: int) -> None:
        # audio_bytes is int16 PCM
        ...

    vad = VadDetector(
        on_utterance=handle_utterance,
        silence_threshold=0.005,
        silence_duration=0.8,
        min_speech=0.15,
        silero_threshold=0.3,
        vad_noise_mult=4.0,
    )

    async def on_audio(chunk: AudioChunk) -> None:
        # chunk.data is float32 LE
        await vad.feed(chunk.data, chunk.sample_rate, chunk.samples)
"""
from __future__ import annotations

import asyncio
import logging
import string
from typing import Awaitable, Callable

import numpy as np

log = logging.getLogger("glasses_agent_nat.vad")

SAMPLE_RATE     = 16_000
_SILERO_WINDOW  = 512       # 32 ms at 16 kHz
_MAX_UTT_S      = 30.0
_PRE_ROLL_CHUNKS = 10       # ~320 ms pre-roll (10 × 32 ms chunks)

_FILLER_PHRASES = frozenset({
    "mm-hmm", "mm hmm", "uh huh", "uh-huh", "uh", "um", "ah", "oh", "eh",
    "huh", "hmm", "yeah", "yep", "yup", "okay", "ok", "right",
    "sure", "thanks", "thank you",
})


def _is_filler(lower: str) -> bool:
    if not lower:
        return True
    tokens = lower.split()
    if len(tokens) == 1:
        return lower.rstrip(string.punctuation) in _FILLER_PHRASES
    if lower in _FILLER_PHRASES:
        return True
    return all(t.rstrip(string.punctuation) in _FILLER_PHRASES for t in tokens)


OnUtteranceCb = Callable[[bytes, int], Awaitable[None]]


class VadDetector:
    """Per-participant Silero VAD + utterance accumulator.

    ``feed()`` accepts raw PCM chunks (int16 bytes at *sample_rate* Hz).
    When a complete utterance is detected (silence after min_speech) the
    on_utterance callback is awaited with the raw int16 PCM bytes and the
    sample rate.
    """

    def __init__(
        self,
        on_utterance:      OnUtteranceCb,
        *,
        silence_threshold: float = 0.005,
        silence_duration:  float = 0.8,
        min_speech:        float = 0.15,
        silero_threshold:  float = 0.3,
        vad_noise_mult:    float = 4.0,
    ) -> None:
        self._on_utterance     = on_utterance
        self._silence_threshold = silence_threshold
        self._silence_duration  = silence_duration
        self._min_speech        = min_speech
        self._silero_threshold  = silero_threshold
        self._vad_noise_mult    = vad_noise_mult

        self._buffer:         list[bytes] = []
        self._buffer_samples: int         = 0
        self._speech_s:       float       = 0.0
        self._silent_s:       float       = 0.0
        self._speaking:       bool        = False
        self._busy:           bool        = False   # on_utterance in flight

        # Rolling pre-roll: keep last N raw PCM chunks before speech onset.
        self._pre_roll: list[bytes] = []

        # Silero VAD (ONNX backend — no GPU or PyTorch required in this venv).
        self._silero      = None
        self._silero_buf  = np.zeros(0, np.float32)
        self._noise_floor = 0.001  # adaptive energy fallback

        try:
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad(onnx=True)
            log.info("Silero VAD loaded (onnx=True)")
        except Exception as exc:
            log.warning(
                "Silero VAD unavailable (%s) — using adaptive energy VAD", exc
            )

    async def feed(self, float32_bytes: bytes, sample_rate: int, n_samples: int) -> None:
        """Process one chunk of float32 LE PCM audio from the XR hub.

        ``float32_bytes`` — raw float32 little-endian bytes (AudioChunk.data).
        ``sample_rate``   — sample rate in Hz.
        ``n_samples``     — number of sample frames in ``float32_bytes``.

        Internally converts to float32 numpy for VAD and to int16 bytes for
        the utterance buffer (the format STT servers expect).
        """
        chunk_s = n_samples / max(sample_rate, 1)

        # Decode float32 → numpy float32 array.
        f32 = np.frombuffer(float32_bytes, dtype=np.float32)

        # Convert to int16 bytes for the utterance buffer (WAV / STT format).
        i16_bytes = (np.clip(f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        # ── speech / silence classification ──────────────────────────────────
        if self._silero is not None:
            try:
                import torch as _torch
                self._silero_buf = np.concatenate([self._silero_buf, f32])
                speech_prob = 0.0
                while len(self._silero_buf) >= _SILERO_WINDOW:
                    window = self._silero_buf[:_SILERO_WINDOW]
                    self._silero_buf = self._silero_buf[_SILERO_WINDOW:]
                    tensor = _torch.from_numpy(np.ascontiguousarray(window))
                    speech_prob = max(
                        speech_prob, float(self._silero(tensor, sample_rate))
                    )
                is_speech = speech_prob > self._silero_threshold
            except Exception:
                # torch unavailable or silero call failed — fall through to energy
                is_speech = self._energy_vad(f32)
        else:
            is_speech = self._energy_vad(f32)

        # ── accumulation (int16 bytes) ────────────────────────────────────────
        if is_speech:
            if not self._speaking:
                log.debug("speech START")
                self._speaking = True
                # Prepend pre-roll so onset of the first word is captured.
                if self._pre_roll:
                    pre = b"".join(self._pre_roll)
                    self._buffer.insert(0, pre)
                    self._buffer_samples += len(pre) // 2
                    self._pre_roll.clear()
            self._buffer.append(i16_bytes)
            self._buffer_samples += n_samples
            self._speech_s += chunk_s
            self._silent_s  = 0.0
        else:
            if self._speaking:
                # Append trailing silence so we don't hard-clip the end.
                self._buffer.append(i16_bytes)
                self._buffer_samples += n_samples
                self._silent_s += chunk_s
            else:
                # Pre-speech: maintain rolling pre-roll window (int16 bytes).
                self._pre_roll.append(i16_bytes)
                if len(self._pre_roll) > _PRE_ROLL_CHUNKS:
                    self._pre_roll.pop(0)

        # ── finalization conditions ────────────────────────────────────────────
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

    def _energy_vad(self, f32: np.ndarray) -> bool:
        """Adaptive energy-based fallback when Silero is unavailable.

        ``f32`` — float32 numpy array of the current chunk.
        """
        rms = float(np.sqrt(np.mean(f32 ** 2))) if len(f32) else 0.0
        if not self._speaking and not self._buffer:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        eff_thr = max(self._silence_threshold, self._noise_floor * self._vad_noise_mult)
        return rms >= eff_thr

    async def _finalize(self, sample_rate: int) -> None:
        if not self._buffer:
            self._speaking       = False
            return
        audio_bytes          = b"".join(self._buffer)
        self._buffer         = []
        self._buffer_samples = 0
        self._speaking       = False
        self._silent_s       = 0.0
        self._speech_s       = 0.0
        self._busy           = True
        # audio_bytes is int16 PCM; compute duration for logging.
        dur_s = (len(audio_bytes) // 2) / max(sample_rate, 1)
        log.info("VAD: utterance finalized  dur=%.2fs", dur_s)
        try:
            await self._on_utterance(audio_bytes, sample_rate)
        except Exception:
            log.exception("on_utterance callback raised")
        finally:
            self._busy = False
