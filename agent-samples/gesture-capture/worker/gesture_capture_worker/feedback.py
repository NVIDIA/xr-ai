# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-owned gesture cues over the public hub return-audio path."""

from __future__ import annotations

import asyncio
import time

import numpy as np
from xr_ai_hub import AudioChunk, ProcessorEndpoint

_RATE = 24000
_CHUNK_S = 0.02
_SAMPLES = int(_RATE * _CHUNK_S)


class GestureFeedback:
    def __init__(self, endpoint: ProcessorEndpoint, participant_id: str, *, enabled: bool, max_gap_s: float) -> None:
        self._endpoint = endpoint
        self._participant_id = participant_id
        self._enabled = enabled
        self._max_gap_s = max_gap_s
        self._until = 0.0
        self._hum_task: asyncio.Task[None] | None = None
        self._closed = False

    async def _send(self, samples: np.ndarray) -> None:
        await asyncio.wait_for(self._endpoint.send_return_audio(AudioChunk(
            participant_id=self._participant_id, track_id="gesture-feedback",
            pts_us=time.time_ns() // 1_000, sample_rate=_RATE, channels=1,
            samples=len(samples), data=samples.astype("<f4").tobytes(),
        )), timeout=self._max_gap_s)

    async def _hum(self) -> None:
        # Four complete 200Hz cycles per chunk avoid discontinuities. Only one
        # short chunk is queued at a time; cancellation cannot retract audio
        # already played, and must not flush unrelated participant audio.
        t = np.arange(_SAMPLES, dtype=np.float32) / _RATE
        samples = 0.025 * np.sin(2 * np.pi * 200 * t)
        while asyncio.get_running_loop().time() < self._until:
            await self._send(samples)
            await asyncio.sleep(_CHUNK_S)

    async def update(self, in_progress: bool) -> None:
        if not self._enabled or self._closed:
            return
        if not in_progress:
            await self.stop()
            return
        self._until = asyncio.get_running_loop().time() + self._max_gap_s
        if self._hum_task is not None and self._hum_task.done():
            task, self._hum_task = self._hum_task, None
            await task
        if self._hum_task is None:
            self._hum_task = asyncio.create_task(self._hum(), name=f"gesture-hum-{self._participant_id}")

    async def recognized(self) -> None:
        if not self._enabled or self._closed:
            return
        await self.stop()
        # Match the wake-word cue's existing 880/1320Hz perfect-fifth pattern,
        # without importing voice-gate's private synthesis helper or voice stack.
        t = np.arange(int(_RATE * 0.25), dtype=np.float32) / _RATE
        samples = (0.55 * np.sin(2 * np.pi * 880 * t) + 0.30 * np.sin(2 * np.pi * 1320 * t))
        samples *= 0.25 * np.exp(-8 * t)
        samples[-120:] *= np.linspace(1, 0, 120, dtype=np.float32)
        for start in range(0, len(samples), _SAMPLES):
            chunk = samples[start:start + _SAMPLES]
            await self._send(chunk)
            await asyncio.sleep(len(chunk) / _RATE)

    async def stop(self) -> None:
        task, self._hum_task = self._hum_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        await self.stop()
