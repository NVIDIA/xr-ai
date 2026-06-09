# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-participant VAD for the glasses-agent-nat worker.

Input is raw int16 little-endian PCM. Silero runs through its ONNX backend when
available; if the local torch/torchaudio stack cannot load, the detector falls
back to the same adaptive energy gate used by glasses-agent.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import numpy as np

log = logging.getLogger("glasses_agent_nat.vad")

_SILERO_SR       = 16_000
_SILERO_WINDOW   = 512
# Hard cap on a single utterance. Only fires when end-of-speech silence is
# never detected — e.g. the energy-VAD fallback in a noisy room, where the
# gate stays above threshold continuously. 30 s let several sentences pile into
# one blob, so STT returned munged multi-sentence garbage ("…what do you see?
# It's really state to day but don't cross over…") and the agent answered the
# wrong question / dropped guidance commands. 15 s still covers any real spoken
# command or demo-step narration (which finalize on the 0.8 s silence gate far
# sooner) while bounding the runaway case to something STT can transcribe.
_MAX_UTT_S       = 15.0
_PRE_ROLL_CHUNKS = 10


OnUtteranceCb   = Callable[[bytes, int], Awaitable[None]]
OnSpeechStartCb = Callable[[], Awaitable[None]]


class VadDetector:
    """Per-participant utterance accumulator with Silero and energy fallback."""

    def __init__(
        self,
        on_utterance:      OnUtteranceCb,
        *,
        on_speech_start:   Optional[OnSpeechStartCb] = None,
        silence_duration:  float = 0.8,
        min_speech:        float = 0.15,
        silero_threshold:  float = 0.5,
        silence_threshold: float = 0.005,
        vad_noise_mult:    float = 4.0,
    ) -> None:
        self._on_utterance       = on_utterance
        self._on_speech_start    = on_speech_start
        self._silence_duration   = silence_duration
        self._min_speech         = min_speech
        self._silero_threshold   = silero_threshold
        self._silence_threshold  = silence_threshold
        self._vad_noise_mult     = vad_noise_mult

        self._buffer:         list[bytes] = []
        self._buffer_samples: int         = 0
        self._speech_s:       float       = 0.0
        self._silent_s:       float       = 0.0
        self._speaking:       bool        = False
        self._speech_start_fired: bool    = False

        self._pre_roll: list[bytes] = []
        self._silero = None
        self._silero_buf = np.zeros(0, np.float32)
        self._noise_floor = 0.001

        try:
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad(onnx=True)
            log.info("Silero VAD loaded (onnx=True)")
        except Exception as exc:
            log.warning("Silero VAD unavailable (%s); using adaptive energy VAD", exc)

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
        if self._silero is not None and hasattr(self._silero, "reset_states"):
            try:
                self._silero.reset_states()
            except Exception:
                log.debug("Silero reset_states failed", exc_info=True)

    async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
        """Process one chunk of int16 little-endian PCM audio."""
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

            if (
                self._on_speech_start is not None
                and not self._speech_start_fired
                and self._speech_s >= self._min_speech
                and prev_speech_s < self._min_speech
            ):
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
            log.info("VAD: max utterance length reached (%.1fs); finalizing", utt_s)
            await self._finalize(sample_rate)
            return

        if (
            self._speaking
            and self._speech_s >= self._min_speech
            and self._silent_s >= self._silence_duration
        ):
            await self._finalize(sample_rate)

    def _classify(self, pcm_int16: bytes, sample_rate: int) -> bool:
        f32 = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        if self._silero is None:
            return self._energy_vad(f32)

        try:
            return self._classify_silero(f32, sample_rate)
        except Exception as exc:
            log.warning("Silero VAD failed (%s); using adaptive energy VAD", exc)
            self._silero = None
            self._silero_buf = np.zeros(0, np.float32)
            return self._energy_vad(f32)

    def _classify_silero(self, f32: np.ndarray, sample_rate: int) -> bool:
        if sample_rate != _SILERO_SR and f32.size:
            n_out = max(1, int(round(f32.size * _SILERO_SR / sample_rate)))
            f32 = np.interp(
                np.linspace(0.0, f32.size - 1, n_out, dtype=np.float32),
                np.arange(f32.size, dtype=np.float32),
                f32,
            ).astype(np.float32)

        import torch

        self._silero_buf = np.concatenate([self._silero_buf, f32])
        speech_prob = 0.0
        while len(self._silero_buf) >= _SILERO_WINDOW:
            window = self._silero_buf[:_SILERO_WINDOW]
            self._silero_buf = self._silero_buf[_SILERO_WINDOW:]
            tensor = torch.from_numpy(np.ascontiguousarray(window))
            speech_prob = max(speech_prob, float(self._silero(tensor, _SILERO_SR)))
        return speech_prob > self._silero_threshold

    def _energy_vad(self, f32: np.ndarray) -> bool:
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
        self._silero_buf = np.zeros(0, np.float32)
        if self._silero is not None and hasattr(self._silero, "reset_states"):
            try:
                self._silero.reset_states()
            except Exception:
                log.debug("Silero reset_states failed", exc_info=True)
        dur_s = (len(audio_bytes) // 2) / max(sample_rate, 1)
        log.info("VAD: utterance finalized  dur=%.2fs", dur_s)
        try:
            await self._on_utterance(audio_bytes, sample_rate)
        except Exception:
            log.exception("on_utterance callback raised")
