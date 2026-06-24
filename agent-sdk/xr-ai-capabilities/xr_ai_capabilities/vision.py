# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable live-camera vision Q&A for agent brains.

``VisionModule`` is the "answer a question about what the camera sees" feature,
factored out of the individual samples (simple-vlm-example, xr-render-demo, …)
that each used to re-implement it. It owns:

  * frame tracking — the latest ``FrameSignal`` per participant, with a
    wall-clock freshness check;
  * camera-on-demand — ``startCamera`` / ``stopCamera`` on the ``clientControl``
    topic, a speculative warmup on the leading edge of speech, and a grace
    timer that keeps the camera on across rapid follow-ups;
  * the VLM streaming call — fetch the freshest frame, encode it, and stream
    answer tokens for downstream sentence-batched TTS (or text output).

A brain builds a ``VisionModule`` when it has a VLM service to back it, and uses
it for any "what do you see"-style query. It exposes two call styles: ``ask``
(streams tokens, for TTS) and ``perceive`` (returns a string, for agentic tool
loops); both share one frame-acquisition + camera-on-demand path.

The module is framework-agnostic: it talks to the hub through a
``ProcessorEndpoint`` (subscribing to ``FrameSignal`` events, fetching frames,
sending ``clientControl`` data, and setting agent status) and has no dependency
on pipecat. A pipecat brain passes ``transport.endpoint``; a non-pipecat agent
passes its own endpoint.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import AsyncIterator

from loguru import logger
from xr_ai_agent import DataMessage, FrameSignal, ProcessorEndpoint
from xr_ai_models import VLMService

from .pixels import encode_image, frame_to_pil


def _now_us() -> int:
    return time.time_ns() // 1_000


class VisionUnavailable(Exception):
    """Raised by :meth:`VisionModule.perceive` when a live frame can't be turned
    into a VLM answer (no camera frame, frame fetch failed, or the VLM errored).
    The message is a short, user-facing sentence suitable to speak."""


class VisionModule:
    """Live-camera VLM question answering, with camera-on-demand.

    Parameters
    ----------
    endpoint:
        The ``ProcessorEndpoint`` to talk to the hub through; the module
        subscribes to frame signals, fetches frames, sends ``clientControl``
        messages, and sets agent status on it. A pipecat brain passes
        ``transport.endpoint``.
    vlm:
        A ``VLMService`` (its ``stream`` is used for token-by-token answers).
    system_prompt:
        Default system prompt for the VLM (overridable per ``ask``).
    frame_max_age_s / camera_on_timeout_s / camera_grace_s:
        Freshness threshold, how long to wait for a fresh frame after
        ``startCamera``, and how long to keep the camera on after a query.
    """

    def __init__(
        self,
        endpoint: ProcessorEndpoint,
        vlm: VLMService,
        *,
        system_prompt: str = "",
        frame_max_age_s: float = 2.0,
        camera_on_timeout_s: float = 15.0,
        camera_grace_s: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._vlm = vlm
        self._system_prompt = system_prompt
        self._frame_max_age_us  = int(frame_max_age_s * 1_000_000)
        self._camera_on_timeout = camera_on_timeout_s
        self._camera_grace_s    = camera_grace_s

        self._latest: dict[tuple[str, str], FrameSignal] = {}
        self._frame_events: dict[str, asyncio.Event] = {}
        self._camera_on: dict[str, bool] = {}
        self._camera_held: set[str] = set()
        self._camera_off_timers: dict[str, asyncio.Task] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def register(self) -> None:
        """Subscribe to the endpoint's frame signals. Call once at setup."""
        self._endpoint.on_frame(self._on_frame)

    async def warmup(self, pid: str) -> None:
        """Speculative camera-on at the leading edge of speech, so the frame is
        usually ready by the time the query reaches the VLM."""
        old = self._camera_off_timers.pop(pid, None)
        if old and not old.done():
            old.cancel()
        try:
            await self._ensure_camera_on(pid)
        except Exception:
            logger.exception("camera warmup failed pid={!r}", pid)

    def release(self, pid: str) -> None:
        """Drop all per-participant state (call from ``on_participant_left``)."""
        self._latest = {k: v for k, v in self._latest.items() if k[0] != pid}
        self._frame_events.pop(pid, None)
        self._camera_on.pop(pid, None)
        self._camera_held.discard(pid)
        timer = self._camera_off_timers.pop(pid, None)
        if timer and not timer.done():
            timer.cancel()

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

    async def _wait_for_camera_frame(self, pid: str, timeout: float) -> FrameSignal | None:
        """Wait up to ``timeout`` for a fresh ``FrameSignal``. Only fresh
        signals are accepted — a stale one from a stopped track would make
        ``request_frame`` deliver an 8x8 placeholder."""
        ev = self._frame_events.setdefault(pid, asyncio.Event())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
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

    # ── camera on demand ───────────────────────────────────────────────────────

    async def _client_control(self, pid: str, action: str) -> None:
        await self._endpoint.send_return_data(DataMessage(
            participant_id = pid,
            topic          = "clientControl",
            pts_us         = _now_us(),
            data           = json.dumps({"action": action}).encode(),
        ))

    async def _ensure_camera_on(self, pid: str) -> None:
        """Send ``startCamera`` once (idempotent); claim the flag before the
        first await so concurrent callers can't double-send."""
        if self._camera_on.get(pid, False):
            return
        self._camera_on[pid] = True
        try:
            logger.info("camera.on → pid={!r}", pid)
            await self._client_control(pid, "startCamera")
        except Exception:
            self._camera_on[pid] = False
            raise

    def _schedule_camera_off(self, pid: str) -> None:
        old = self._camera_off_timers.pop(pid, None)
        if old and not old.done():
            old.cancel()

        async def _off() -> None:
            try:
                await asyncio.sleep(self._camera_grace_s)
                if pid not in self._camera_held:
                    self._camera_on[pid] = False
                    await self._client_control(pid, "stopCamera")
            except asyncio.CancelledError:
                # Expected: a newer query cancels this grace timer before it
                # fires (see the cancel above), keeping the camera on. Nothing
                # to do — the timer simply doesn't stop the camera.
                pass

        self._camera_off_timers[pid] = asyncio.create_task(_off())

    @contextlib.asynccontextmanager
    async def _camera_session(self, pid: str):
        """Bracket one query's camera use: cancel any pending stop and mark the
        participant held on entry; release the hold + reschedule stop on exit.

        Shared by ``ask`` and ``perceive`` (their identical preamble/teardown).
        Logs per-call elapsed time so VLM latency stays visible in the logs even
        though the SDK emits no sample-level task banner.
        """
        old = self._camera_off_timers.pop(pid, None)
        if old and not old.done():
            old.cancel()
        self._camera_held.add(pid)
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._camera_held.discard(pid)
            self._schedule_camera_off(pid)
            logger.info("vision call pid={!r} elapsed={:.2f}s",
                        pid, time.monotonic() - t0)

    # ── frame acquisition (shared by ask + perceive) ───────────────────────────

    async def _acquire_image_url(self, pid: str) -> str:
        """Ensure the camera is on, wait for a fresh frame, fetch + encode it to
        a JPEG data URL. Raises :class:`VisionUnavailable` if no usable frame."""
        sig = self._latest_signal(pid)
        if not (sig and self._is_fresh(sig)):
            await self._ensure_camera_on(pid)
            sig = await self._wait_for_camera_frame(pid, self._camera_on_timeout)
            if sig is None:
                self._camera_on[pid] = False
                raise VisionUnavailable("Camera unavailable, please try again.")
        frame = await self._endpoint.request_frame(sig)
        if frame is None:
            raise VisionUnavailable("Frame data unavailable — please retry.")
        loop = asyncio.get_running_loop()
        image_url = await loop.run_in_executor(
            None, lambda: encode_image(frame_to_pil(frame)),
        )
        logger.info("vision  pid={!r}  {}x{}", pid, frame.width, frame.height)
        return image_url

    # ── the VLM call: streaming (ask) and one-shot (perceive) ───────────────────

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
        async with self._camera_session(pid):
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

    async def perceive(
        self, pid: str, query: str, *, system_prompt: str | None = None,
    ) -> str:
        """Acquire a fresh frame and return the VLM answer as a **string**
        (one-shot). Use this from an agentic tool loop that needs a value to feed
        back to the LLM rather than a token stream for TTS.

        Raises :class:`VisionUnavailable` (with a speakable message) on no
        camera/frame, VLM error, or an empty answer.

        Status contract: unlike ``ask``, ``perceive`` does **not** touch the
        agent-status badge — the calling agentic loop owns its own status (it
        is typically mid-turn doing other work), so this method stays out of it.
        """
        async with self._camera_session(pid):
            image_url = await self._acquire_image_url(pid)   # raises VisionUnavailable
            try:
                resp = await self._vlm.ask_image(
                    image_url, query, system_prompt=system_prompt or self._system_prompt,
                )
            except Exception as exc:
                logger.error("vlm-server error: {}", exc)
                raise VisionUnavailable("VLM server unavailable — please retry.") from exc
            answer = (resp.content or "").strip()
            if not answer:
                raise VisionUnavailable("I couldn't make out anything in the view.")
            return answer
