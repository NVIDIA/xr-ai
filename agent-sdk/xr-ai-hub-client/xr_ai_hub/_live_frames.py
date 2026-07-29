# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped access to fresh raw camera frames."""
from __future__ import annotations

import asyncio
import time

from ._processor import ProcessorEndpoint
from ._types import FrameData, FrameSignal, ParticipantEvent


class FrameUnavailable(Exception):
    """Raised when a fresh camera frame cannot be obtained before the timeout."""


class LiveFrameSource:
    """Track frame signals and fetch fresh pixels through ``ProcessorEndpoint``.

    The source stops at raw ``FrameData`` so consumers own image conversion and
    model work without adding dependencies to the hub client.
    """

    def __init__(
        self,
        endpoint: ProcessorEndpoint,
        *,
        max_age_s: float = 2.0,
        timeout_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._max_age_us = int(max_age_s * 1_000_000)
        self._timeout_s = timeout_s
        self._latest: dict[tuple[str, str], FrameSignal] = {}
        self._events: dict[str, set[asyncio.Event]] = {}
        endpoint.on_frame(self._on_frame)
        endpoint.on_participant(self._on_participant)

    async def _on_frame(self, signal: FrameSignal) -> None:
        self._latest[(signal.participant_id, signal.track_id)] = signal
        for event in self._events.get(signal.participant_id, ()):
            event.set()

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            self.release(event.participant_id)

    def _freshest(self, participant_id: str) -> FrameSignal | None:
        candidates = [
            signal
            for (pid, _track), signal in self._latest.items()
            if pid == participant_id
        ]
        return max(candidates, key=lambda signal: signal.pts_us) if candidates else None

    def _is_fresh(self, signal: FrameSignal) -> bool:
        return time.time_ns() // 1_000 - signal.pts_us < self._max_age_us

    def participants(self) -> list[str]:
        """Return participant IDs with a frame inside the configured freshness window."""
        return sorted(
            {
                participant_id
                for (participant_id, _track), signal in self._latest.items()
                if self._is_fresh(signal)
            }
        )

    async def get(self, participant_id: str) -> FrameData:
        """Return fresh pixels for ``participant_id`` or raise ``FrameUnavailable``."""
        signal = self._freshest(participant_id)
        if signal is None or not self._is_fresh(signal):
            event = asyncio.Event()
            self._events.setdefault(participant_id, set()).add(event)
            try:
                deadline = asyncio.get_running_loop().time() + self._timeout_s
                while signal is None or not self._is_fresh(signal):
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise FrameUnavailable("No camera frame available — please try again.")
                    event.clear()
                    # Recheck after clearing so a concurrent frame signal is not missed.
                    signal = self._freshest(participant_id)
                    if signal is not None and self._is_fresh(signal):
                        break
                    try:
                        await asyncio.wait_for(event.wait(), timeout=remaining)
                    except asyncio.TimeoutError as exc:
                        raise FrameUnavailable(
                            "No camera frame available — please try again."
                        ) from exc
                    signal = self._freshest(participant_id)
            finally:
                events = self._events.get(participant_id)
                if events is not None:
                    events.discard(event)
                    if not events:
                        self._events.pop(participant_id, None)

        frame = await self._endpoint.request_frame(signal)
        if frame is None:
            raise FrameUnavailable("Frame data unavailable — please retry.")
        return frame

    def release(self, participant_id: str) -> None:
        """Drop cached signals and wake pending requests for a disconnected participant."""
        self._latest = {
            key: signal
            for key, signal in self._latest.items()
            if key[0] != participant_id
        }
        for event in self._events.get(participant_id, ()):
            event.set()


__all__ = ["FrameUnavailable", "LiveFrameSource"]
