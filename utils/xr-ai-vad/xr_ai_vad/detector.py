# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VadDetector — async Silero VAD with adaptive energy fallback.

Designed for agent workers that ingest microphone audio over the XR
Media Hub IPC layer.  The hub delivers float32 LE PCM via
``AudioChunk.data``; this detector consumes that raw byte format and
emits int16 PCM utterance bytes (the format STT servers expect) via an
async callback when speech ends.

Key features
------------
- Silero VAD (``onnx=True``) with adaptive energy fallback when silero
  or torch is unavailable.
- ~320 ms pre-roll (10 chunks of ~32 ms each) so the attack of the first
  word is captured.
- Optional ``on_speech_start`` callback fires the moment speech crosses
  the ``min_speech`` threshold — useful for speculatively warming up
  downstream resources (camera, model server, …) before the user has
  finished speaking.
- Hard cap of 30 s per utterance.

Usage::

    async def handle_utterance(audio_bytes: bytes, sample_rate: int) -> None:
        # audio_bytes is int16 PCM, ready for WAV / STT.
        ...

    async def warm_up() -> None:
        # Fires once per utterance when speech_s first crosses min_speech.
        ...

    vad = VadDetector(
        on_utterance    = handle_utterance,
        on_speech_start = warm_up,
        silero_threshold= 0.5,
    )

    async def on_audio(chunk: AudioChunk) -> None:
        await vad.feed(chunk.data, chunk.sample_rate, chunk.samples)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import numpy as np

log = logging.getLogger("xr_ai_vad")

_SILERO_WINDOW   = 512    # 32 ms at 16 kHz
_MAX_UTT_S       = 30.0
_PRE_ROLL_CHUNKS = 10     # ~320 ms pre-roll (10 × 32 ms)


OnUtteranceCb   = Callable[[bytes, int], Awaitable[None]]
OnSpeechStartCb = Callable[[], Awaitable[None]]


class VadDetector:
    """Per-participant Silero VAD + utterance accumulator.

    ``feed()`` accepts raw float32 LE PCM bytes (``AudioChunk.data``).
    When a complete utterance is detected (silence after ``min_speech``)
    the ``on_utterance`` callback is awaited with int16 PCM bytes and
    the sample rate.

    When ``on_speech_start`` is provided, it fires once per utterance at
    the moment ``speech_s`` first crosses ``min_speech``.  This is a
    "leading edge" hook intended for speculative work (e.g. warming up
    a downstream resource before STT completes).
    """

    def __init__(
        self,
        on_utterance:      OnUtteranceCb,
        *,
        on_speech_start:   Optional[OnSpeechStartCb] = None,
        silence_threshold: float = 0.005,
        silence_duration:  float = 0.8,
        min_speech:        float = 0.15,
        silero_threshold:  float = 0.5,
        vad_noise_mult:    float = 4.0,
    ) -> None:
        self._on_utterance      = on_utterance
        self._on_speech_start   = on_speech_start
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
        # One-shot per utterance: ensures on_speech_start fires at most
        # once between finalizations.
        self._speech_start_fired: bool = False

        # Rolling pre-roll: keep last N raw int16 chunks before speech onset.
        self._pre_roll: list[bytes] = []

        # Silero state — onnx backend so no GPU/torch is required at runtime.
        self._silero      = None
        self._silero_buf  = np.zeros(0, np.float32)
        self._noise_floor = 0.001  # adaptive energy fallback baseline

        try:
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad(onnx=True)
            log.info("Silero VAD loaded (onnx=True)")
        except Exception as exc:
            log.warning(
                "Silero VAD unavailable (%s) — using adaptive energy VAD", exc
            )

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

    async def feed(self, float32_bytes: bytes, sample_rate: int, n_samples: int) -> None:
        """Process one chunk of float32 LE PCM audio.

        ``float32_bytes`` — raw float32 little-endian bytes.
        ``sample_rate``   — sample rate in Hz.
        ``n_samples``     — number of sample frames in ``float32_bytes``.
        """
        chunk_s = n_samples / max(sample_rate, 1)

        f32       = np.frombuffer(float32_bytes, dtype=np.float32)
        i16_bytes = (np.clip(f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        is_speech = self._classify(f32, sample_rate)

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
            prev_speech_s = self._speech_s
            self._speech_s += chunk_s
            self._silent_s  = 0.0

            # Leading-edge hook: fire once when we first cross min_speech.
            if (self._on_speech_start is not None
                    and not self._speech_start_fired
                    and self._speech_s >= self._min_speech
                    and prev_speech_s < self._min_speech):
                self._speech_start_fired = True
                asyncio.create_task(self._safe_speech_start())
        else:
            if self._speaking:
                self._buffer.append(i16_bytes)
                self._buffer_samples += n_samples
                self._silent_s += chunk_s
            else:
                self._pre_roll.append(i16_bytes)
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

    def _classify(self, f32: np.ndarray, sample_rate: int) -> bool:
        """Return True if the chunk is speech."""
        if self._silero is None:
            return self._energy_vad(f32)
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
            return speech_prob > self._silero_threshold
        except Exception:
            # torch unavailable or silero call failed — fall through to energy.
            return self._energy_vad(f32)

    def _energy_vad(self, f32: np.ndarray) -> bool:
        """Adaptive energy fallback used when Silero is unavailable."""
        rms = float(np.sqrt(np.mean(f32 ** 2))) if len(f32) else 0.0
        if not self._speaking and not self._buffer:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        eff_thr = max(self._silence_threshold, self._noise_floor * self._vad_noise_mult)
        return rms >= eff_thr

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
