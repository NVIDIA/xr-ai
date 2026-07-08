# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live, freshness-aware per-participant camera frame acquisition.

``LiveFrameSource`` is the "track the latest camera frame per participant and
fetch its pixels on demand" logic, factored out so it has exactly one
implementation in the repo — :class:`~xr_ai_skills.vision.VisionModule` and
``video-mcp`` both build on it, rather than each maintaining their own copy of
frame tracking and freshness/wait semantics.

It owns:

  * frame tracking — the latest ``FrameSignal`` per participant, resolved
    across tracks by wall-clock ``pts_us`` (``seq`` restarts on camera
    restart, so it can't be used to pick the newest track);
  * freshness gating — a cached signal only counts if it's within
    ``frame_max_age_s`` of now;
  * an event-driven wait (up to ``frame_timeout_s``) for a fresh frame to
    arrive, rather than polling or failing instantly;
  * the on-demand pixel fetch (``ProcessorEndpoint.request_frame``) once a
    usable signal is found.

It does not know about pixel formats, VLMs, or MCP — callers decide what to
do with the ``FrameData`` they get back.
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger
from xr_ai_agent import FrameData, FrameSignal, ProcessorEndpoint


def _now_us() -> int:
    return time.time_ns() // 1_000


class LiveFrameUnavailable(Exception):
    """Raised by :meth:`LiveFrameSource.get_frame` when no usable frame could
    be acquired (no signal within the timeout, or the frame fetch failed).
    The message is a short, user-facing sentence suitable to speak."""


class LiveFrameSource:
    """Tracks the latest live camera frame per participant and fetches pixels
    on demand.

    Parameters
    ----------
    endpoint:
        The ``ProcessorEndpoint`` to talk to the hub through; this source
        subscribes to frame signals and fetches frames on demand.
    frame_max_age_s:
        Maximum age of a cached frame signal before it is considered stale.
    frame_timeout_s:
        How long to wait for a fresh frame before raising
        :class:`LiveFrameUnavailable`.
    """

    def __init__(
        self,
        endpoint: ProcessorEndpoint,
        *,
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._frame_max_age_us = int(frame_max_age_s * 1_000_000)
        self._frame_timeout_s  = frame_timeout_s

        self._latest: dict[tuple[str, str], FrameSignal] = {}
        self._frame_events: dict[str, asyncio.Event] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to the endpoint's frame signals. Call once at setup."""
        self._endpoint.on_frame(self._on_frame)

    def release(self, pid: str) -> None:
        """Drop all per-participant state (call from ``on_participant_left``)."""
        self._latest = {k: v for k, v in self._latest.items() if k[0] != pid}
        self._frame_events.pop(pid, None)

    @property
    def connected_participants(self) -> frozenset[str]:
        """Raw identities currently connected to the hub (live IPC roster)."""
        return self._endpoint.connected_participants

    # ── frame tracking ─────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        prev = self._latest.get((sig.participant_id, sig.track_id))
        self._latest[(sig.participant_id, sig.track_id)] = sig
        if prev is None:
            logger.info(
                "first frame signal  pid={!r}  track={}  age_ms={:.0f}",
                sig.participant_id, sig.track_id,
                (_now_us() - sig.pts_us) / 1_000,
            )
        ev = self._frame_events.get(sig.participant_id)
        if ev is not None:
            ev.set()

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        # pts_us is wall-clock; seq restarts on each camera restart so it would
        # pick a stale track's last entry.
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        return max(candidates, key=lambda s: s.pts_us) if candidates else None

    def _is_fresh(self, sig: FrameSignal) -> bool:
        return _now_us() - sig.pts_us < self._frame_max_age_us

    async def _wait_for_frame(self, pid: str) -> FrameSignal | None:
        """Wait up to ``frame_timeout_s`` for a fresh ``FrameSignal``."""
        ev = self._frame_events.setdefault(pid, asyncio.Event())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._frame_timeout_s
        ev.clear()
        sig = self._latest_signal(pid)
        if sig is not None and self._is_fresh(sig):
            return sig
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(ev.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                ev.clear()
                continue
            sig = self._latest_signal(pid)
            if sig is not None and self._is_fresh(sig):
                return sig
            ev.clear()

    # ── frame acquisition ──────────────────────────────────────────────────────

    async def get_frame(self, pid: str) -> FrameData:
        """Wait for a fresh frame and fetch its pixel data.

        Raises :class:`LiveFrameUnavailable` if no usable frame arrives
        within ``frame_timeout_s``, or the fetch itself fails.
        """
        sig = self._latest_signal(pid)
        if not (sig and self._is_fresh(sig)):
            sig = await self._wait_for_frame(pid)
            if sig is None:
                raise LiveFrameUnavailable("No camera frame available — please try again.")
        frame = await self._endpoint.request_frame(sig)
        if frame is None:
            raise LiveFrameUnavailable("Frame data unavailable — please retry.")
        logger.info("frame  pid={!r}  {}x{}", pid, frame.width, frame.height)
        return frame
