# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-frame acquisition used by native vision functions."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ._images import frame_jpeg_data_url


class LiveFrameUnavailable(RuntimeError):
    """Raised when no usable live camera frame can be acquired."""


class LiveFrameSource:
    """Track frame signals and fetch the freshest frame for each participant."""

    def __init__(self, endpoint: Any, *, max_age_s: float, timeout_s: float) -> None:
        self._endpoint = endpoint
        self._max_age_us = int(max_age_s * 1_000_000)
        self._timeout_s = timeout_s
        self._latest: dict[tuple[str, str], Any] = {}
        self._events: dict[str, asyncio.Event] = {}
        endpoint.on_frame(self._on_frame)

    async def _on_frame(self, signal: Any) -> None:
        self._latest[(signal.participant_id, signal.track_id)] = signal
        event = self._events.get(signal.participant_id)
        if event is not None:
            event.set()

    def release(self, participant_id: str) -> None:
        self._latest = {key: signal for key, signal in self._latest.items() if key[0] != participant_id}
        self._events.pop(participant_id, None)

    async def image_url(self, participant_id: str) -> str:
        signal = self._freshest(participant_id)
        if signal is None or not self._is_fresh(signal):
            signal = await self._wait_for_fresh_signal(participant_id)
        if signal is None:
            raise LiveFrameUnavailable("No camera frame available — please try again.")

        frame = await self._endpoint.request_frame(signal)
        if frame is None:
            raise LiveFrameUnavailable("Frame data unavailable — please retry.")
        return await asyncio.to_thread(frame_jpeg_data_url, frame)

    def _freshest(self, participant_id: str):
        signals = [signal for key, signal in self._latest.items() if key[0] == participant_id]
        return max(signals, key=lambda signal: signal.pts_us) if signals else None

    def _is_fresh(self, signal: Any) -> bool:
        return time.time_ns() // 1_000 - signal.pts_us < self._max_age_us

    async def _wait_for_fresh_signal(self, participant_id: str):
        event = self._events.setdefault(participant_id, asyncio.Event())
        deadline = asyncio.get_running_loop().time() + self._timeout_s
        event.clear()
        while True:
            signal = self._freshest(participant_id)
            if signal is not None and self._is_fresh(signal):
                return signal
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            event.clear()
