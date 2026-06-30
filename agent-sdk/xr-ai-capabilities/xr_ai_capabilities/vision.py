# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable live-camera vision Q&A for agent brains.

``VisionModule`` is the "answer a question about what the camera sees" feature,
factored out of the individual samples (simple-vlm-example, xr-render-demo, …)
that each used to re-implement it. It owns:

  * frame tracking — the latest ``FrameSignal`` per participant, with a
    wall-clock freshness check;
  * the VLM call — fetch the freshest frame, encode it, and stream answer
    tokens for downstream sentence-batched TTS (or text output).

Camera streaming is always-on (the client streams continuously); this module
never sends ``startCamera`` / ``stopCamera`` control messages.

A brain builds a ``VisionModule`` when it has a VLM service to back it, and uses
it for any "what do you see"-style query. It exposes two call styles: ``ask``
(streams tokens, for TTS) and ``perceive`` (returns a string, for agentic tool
loops); both share one frame-acquisition path.

The module is framework-agnostic: it talks to the hub through a
``ProcessorEndpoint`` (subscribing to ``FrameSignal`` events, fetching frames,
and setting agent status) and has no dependency on pipecat. A pipecat brain
passes ``transport.endpoint``; a non-pipecat agent passes its own endpoint.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from loguru import logger
from xr_ai_agent import FrameSignal, ProcessorEndpoint
from xr_ai_models import VLMService

from .pixels import encode_image, frame_to_pil


def _now_us() -> int:
    return time.time_ns() // 1_000


class VisionUnavailable(Exception):
    """Raised by :meth:`VisionModule.perceive` when a live frame can't be turned
    into a VLM answer (no frame available, frame fetch failed, or VLM errored).
    The message is a short, user-facing sentence suitable to speak."""


class VisionModule:
    """Live-camera VLM question answering.

    Camera streaming is always-on — this module does not send camera control
    messages.  It waits up to ``frame_timeout_s`` for a fresh frame to arrive
    before raising :class:`VisionUnavailable`.

    Parameters
    ----------
    endpoint:
        The ``ProcessorEndpoint`` to talk to the hub through; the module
        subscribes to frame signals, fetches frames, and sets agent status.
        A pipecat brain passes ``transport.endpoint``.
    vlm:
        A ``VLMService`` (its ``stream`` is used for token-by-token answers).
    system_prompt:
        Default system prompt for the VLM (overridable per ``ask``).
    frame_max_age_s:
        Maximum age of a cached frame signal before it is considered stale.
    frame_timeout_s:
        How long to wait for a fresh frame before raising
        :class:`VisionUnavailable`.
    """

    def __init__(
        self,
        endpoint: ProcessorEndpoint,
        vlm: VLMService,
        *,
        system_prompt: str = "",
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._vlm = vlm
        self._system_prompt = system_prompt
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

    async def _acquire_image_url(self, pid: str) -> str:
        """Wait for a fresh frame, fetch and encode it to a JPEG data URL.
        Raises :class:`VisionUnavailable` if no usable frame arrives in time."""
        sig = self._latest_signal(pid)
        if not (sig and self._is_fresh(sig)):
            sig = await self._wait_for_frame(pid)
            if sig is None:
                raise VisionUnavailable("No camera frame available — please try again.")
        frame = await self._endpoint.request_frame(sig)
        if frame is None:
            raise VisionUnavailable("Frame data unavailable — please retry.")
        loop = asyncio.get_running_loop()
        image_url = await loop.run_in_executor(
            None, lambda: encode_image(frame_to_pil(frame)),
        )
        logger.info("vision  pid={!r}  {}x{}", pid, frame.width, frame.height)
        return image_url

    # ── the VLM call: streaming (ask) and one-shot (perceive) ──────────────────

    async def ask(
        self, pid: str, query: str, *, system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Acquire a fresh frame and **stream** VLM answer tokens (for TTS).

        On a failure the user should hear, yields a single canned line and
        returns — downstream TTS / text output handles it like any answer.

        Status contract: ``ask`` drives the agent-status badge — it sets
        ``"processing"`` for the duration of the VLM stream and ``"idle"``
        after. (``perceive`` deliberately does *not*; see its note.)
        """
        t0 = time.monotonic()
        try:
            image_url = await self._acquire_image_url(pid)
        except VisionUnavailable as exc:
            yield str(exc)
            return
        await self._endpoint.set_status("processing", pid)
        try:
            async for token in self._vlm.stream(
                image_url, query, system_prompt=system_prompt or self._system_prompt,
            ):
                yield token
        except Exception as exc:
            logger.error("vlm-server error: {}", exc)
            yield "VLM server unavailable — please retry."
            return
        finally:
            await self._endpoint.set_status("idle", pid)
            logger.info("vision call pid={!r} elapsed={:.2f}s", pid, time.monotonic() - t0)

    async def perceive(
        self, pid: str, query: str, *, system_prompt: str | None = None,
    ) -> str:
        """Acquire a fresh frame and return the VLM answer as a **string**
        (one-shot). Use this from an agentic tool loop that needs a value to feed
        back to the LLM rather than a token stream for TTS.

        Raises :class:`VisionUnavailable` (with a speakable message) on no
        frame, VLM error, or an empty answer.

        Status contract: unlike ``ask``, ``perceive`` does **not** touch the
        agent-status badge — the calling agentic loop owns its own status (it
        is typically mid-turn doing other work), so this method stays out of it.
        """
        t0 = time.monotonic()
        image_url = await self._acquire_image_url(pid)   # raises VisionUnavailable
        try:
            resp = await self._vlm.ask_image(
                image_url, query, system_prompt=system_prompt or self._system_prompt,
            )
        except Exception as exc:
            logger.error("vlm-server error: {}", exc)
            raise VisionUnavailable("VLM server unavailable — please retry.") from exc
        finally:
            logger.info("vision call pid={!r} elapsed={:.2f}s", pid, time.monotonic() - t0)
        answer = (resp.content or "").strip()
        if not answer:
            raise VisionUnavailable("I couldn't make out anything in the view.")
        return answer
