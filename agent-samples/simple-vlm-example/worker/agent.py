# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmBrain — the sample-specific bits that aren't the conversation loop.

Owns camera-on-demand, frame tracking, the per-query VLM call, and the
data-channel ("ping" / text query) handler.  Everything else
(VAD/STT/voice-gate/streaming-TTS/cancel-flush) lives in
``xr_ai_conversation.ConversationLoop`` — the worker constructs the brain,
hands its callbacks to the loop, and gives the brain a reference to
``loop.dispatch`` so the data-channel path can dispatch through the same
cancel/flush pipeline.

Camera on demand
----------------
The agent periodically sends ``{"action":"stopCamera"}`` on the
``clientControl`` topic to every connected participant.  Clients in
"always-on" camera mode ignore this signal; clients in "camera on demand"
mode honour it and stop streaming.

When a query needs a video frame and the latest is stale (or absent), the
agent sends ``{"action":"startCamera"}`` and waits up to
``camera_on_timeout_s`` for a fresh frame before proceeding.  While a
query is actively using the camera the periodic stop is suppressed so
rapid follow-up queries don't cause a stop/start cycle.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Awaitable, Callable

import httpx
from loguru import logger
from xr_ai_agent import (DataMessage, FrameSignal, ProcessorEndpoint)
from xr_ai_conversation import now_us
from xr_ai_logging import print_task_done_banner
from xr_ai_models import VLMService

from pixels import encode_image, frame_to_pil


DEFAULT_SYSTEM_PROMPT = (
    "You are an XR assistant. You can see the user's live camera feed, "
    "but you are not required to use it. Decide per question:\n"
    "- If the question is about what is visible (e.g. 'what am I looking "
    "at', 'what does this say', 'is the door open', 'describe this', "
    "'what color is the X'), answer from the image.\n"
    "- If the question is general knowledge, a definition, a calculation, "
    "a chat, or anything not tied to the scene (e.g. 'what's the capital "
    "of France', 'tell me a joke', 'explain entropy', 'how do I boil "
    "pasta'), answer like a normal assistant and ignore the image.\n"
    "- When it is ambiguous, prefer the visual answer if the camera shows "
    "something obviously relevant; otherwise answer generally.\n"
    "\n"
    "Style:\n"
    "- Speak directly to me in second person where natural: 'You are looking "
    "at…', 'I can see…'. Never refer to 'the user' in the third person.\n"
    "- Reply in plain conversational English — never JSON, code, or markdown.\n"
    "- Keep replies to 10-15 words by default. Only go longer when I "
    "explicitly ask for detail (e.g. 'describe in detail', 'tell me more', "
    "'elaborate', 'explain').\n"
    "- If I say 'stop', ask you to be quiet, or ask you to stop "
    "talking, just acknowledge briefly with something like 'Okay, I will stop.' "
    "and say nothing else."
)


# Signature of ``ConversationLoop.dispatch`` — passed in after construction
# so the data-channel handler can route through the same cancel/flush path
# as the voice gate.
DispatchFn = Callable[..., Awaitable[None]]


class SimpleVlmBrain:
    """Sample-specific callbacks for ``ConversationLoop``.

    Holds the camera-on-demand state machine, the latest frame per
    participant, and the per-query VLM call.  ``handle_query`` is the
    ``on_query`` callback wired into the loop; ``on_data`` is registered
    on the ``ProcessorEndpoint`` to route data-channel text through
    ``loop.dispatch``."""

    def __init__(
        self,
        ep:  ProcessorEndpoint,
        vlm: VLMService,
        *,
        default_prompt:      str   = "Describe what you see.",
        system_prompt:       str   = DEFAULT_SYSTEM_PROMPT,
        frame_max_age_s:     float = 2.0,
        camera_on_timeout_s: float = 15.0,
        camera_grace_s:      float = 5.0,
    ) -> None:
        self._ep  = ep
        self._vlm = vlm

        self._default_prompt    = default_prompt
        self._system_prompt     = system_prompt
        self._frame_max_age_us  = int(frame_max_age_s * 1_000_000)
        self._camera_on_timeout = camera_on_timeout_s
        self._camera_grace_s    = camera_grace_s

        # Frame tracking.
        self._latest:       dict[tuple[str, str], FrameSignal] = {}
        self._frame_events: dict[str, asyncio.Event]           = {}

        # Camera on demand state.
        self._camera_on:         dict[str, bool]           = {}  # pid → agent requested camera on
        self._camera_held:       set[str]                  = set()  # pids in active query
        self._camera_off_timers: dict[str, asyncio.Task]   = {}  # pid → delayed-off task

        # ``ConversationLoop.dispatch``; assigned by the worker after both
        # the brain and the loop are constructed (chicken-and-egg).
        self.dispatch: DispatchFn | None = None

    # ── data path: text → query (with "ping" → default prompt) ────────────────

    async def on_data(self, msg: DataMessage) -> None:
        """Route data-channel text through ``loop.dispatch`` so it shares
        the cancel/flush pipeline with the voice path."""
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        logger.info("data query  pid={!r} (text suppressed)", msg.participant_id)
        logger.debug("data query  pid={!r}  {!r}", msg.participant_id, text[:80])
        if self.dispatch is None:
            logger.error("brain.dispatch not wired; dropping data query pid={!r}",
                         msg.participant_id)
            return
        await self.dispatch(msg.participant_id, text, pts_us=msg.pts_us)

    # ── speech-start hook (speculative camera warmup) ─────────────────────────

    async def on_speech_start(self, pid: str) -> None:
        """Speculative camera warmup the moment speech crosses min_speech.

        Fires while the user is still talking — by the time STT finishes,
        the camera is usually already streaming."""
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()
        await self._ensure_camera_on(pid)

    # ── stop / phrase-only / drop hooks (extra camera-off scheduling) ─────────

    async def on_stop_extra(self, pid: str) -> None:
        """Schedule camera-off when the user says stop."""
        self._schedule_camera_off(pid)

    async def on_phrase_only(self, pid: str) -> None:
        """Schedule camera-off when the magic phrase arrives without a payload
        and the follow-up window opens, in case the user never follows up."""
        self._schedule_camera_off(pid)

    async def on_drop(self, pid: str, text: str) -> None:
        """Schedule camera-off when the gate drops an utterance so the
        speculative camera-on from ``on_speech_start`` doesn't leak."""
        self._schedule_camera_off(pid)

    # ── on_query: the actual VLM call ─────────────────────────────────────────

    async def handle_query(
        self, pid: str, text: str, fresh_match: bool,
    ) -> str | AsyncIterator[str]:
        """Answer ``text`` against the latest video frame for ``pid``.

        Returns a canned string for fast-fail paths (camera unavailable,
        frame data unavailable) or an async iterator that yields VLM
        tokens.  In both forms the brain owns the camera-on-demand
        bookkeeping; the loop owns sentence-batched TTS + cancel/flush."""
        query = self._default_prompt if text.lower().strip() == "ping" else text

        # Cancel any pending camera-off so a rapid follow-up query doesn't
        # see the camera turn off between the previous grace period firing.
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()

        self._camera_held.add(pid)
        t0 = time.monotonic()

        def _finish(status: str) -> None:
            self._camera_held.discard(pid)
            self._schedule_camera_off(pid)
            print_task_done_banner(
                "simple-vlm-example",
                status=status,
                detail=f"pid={pid!r}  query={query[:60]!r}",
                duration_s=time.monotonic() - t0,
            )

        try:
            # Acquire a fresh frame, requesting the camera if needed.
            sig = self._latest_signal(pid)
            if not (sig and self._is_fresh(sig)):
                await self._ensure_camera_on(pid)
                sig = await self._wait_for_camera_frame(pid, self._camera_on_timeout)
                if sig is None:
                    # Reset so the next query re-sends startCamera rather than
                    # treating the camera as already on when it never delivered frames.
                    self._camera_on[pid] = False
                    _finish("done")
                    return "Camera unavailable, please try again."

            frame = await self._ep.request_frame(sig)
            if frame is None:
                _finish("done")
                return "Frame data unavailable — please retry."
        except asyncio.CancelledError:
            _finish("interrupted")
            raise
        except Exception:
            _finish("error")
            raise

        image_url = encode_image(frame_to_pil(frame))
        logger.info(
            "vlm  pid={!r}  {}x{}  query={!r}",
            pid, frame.width, frame.height, query[:60],
        )

        # Ownership of _finish passes to the streaming generator; it runs
        # exactly once, either on natural completion or via the loop's
        # cancel (which propagates into the generator at its yield point).
        return self._stream_vlm(pid, image_url, query, _finish)

    async def _stream_vlm(
        self, pid: str, image_url: str, query: str,
        finish: Callable[[str], None],
    ) -> AsyncIterator[str]:
        """Inner generator: stream VLM tokens with status/finish bookkeeping.

        The outer ``handle_query`` already acquired the frame; this
        iterator owns the status flag, the streaming HTTP call, and the
        per-query banner.  The loop's cancel propagates here via
        ``aclose()`` so cancellation runs the ``finally`` block."""
        await self._ep.set_status("processing", pid)
        status = "done"
        emitted_any = False
        try:
            try:
                async for token in self._vlm.stream(
                    image_url, query, system_prompt=self._system_prompt,
                ):
                    emitted_any = True
                    yield token
            except httpx.HTTPError as exc:
                logger.error("vlm-server error: {}", exc)
                # Original behaviour: surface a canned "VLM server
                # unavailable" message.  Only safe to yield it when we
                # haven't yielded anything else — otherwise the user
                # would hear truncated VLM output followed by the canned
                # message glued together.
                if not emitted_any:
                    yield "VLM server unavailable — please retry."
                # else: log + drop, the partial reply already went out.
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except GeneratorExit:
            # aclose() path — the loop's outer task was cancelled while
            # not suspended in our __anext__, so the gc/aclose closes us.
            status = "interrupted"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            try:
                await self._ep.set_status("idle", pid)
            except Exception:
                logger.exception("set_status idle failed pid={!r}", pid)
            finish(status)

    # ── camera on demand ──────────────────────────────────────────────────────

    async def _client_control(self, pid: str, action: str) -> None:
        """Send a camera-control signal on the ``clientControl`` topic."""
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="clientControl",
            pts_us=now_us(),
            data=json.dumps({"action": action}).encode(),
        ))

    async def _ensure_camera_on(self, pid: str) -> None:
        """Send startCamera if we haven't already (idempotent)."""
        if not self._camera_on.get(pid, False):
            # Claim the flag before the first await so concurrent callers
            # (speculative on_speech_start + handle_query) can't both see
            # False and each send startCamera.
            self._camera_on[pid] = True
            try:
                logger.info("camera.on → pid={!r}", pid)
                await self._client_control(pid, "startCamera")
            except Exception:
                self._camera_on[pid] = False  # rollback so next call retries
                raise

    async def _wait_for_camera_frame(
        self, pid: str, timeout: float,
    ) -> FrameSignal | None:
        """Wait up to ``timeout`` seconds for a fresh FrameSignal for ``pid``.

        We only accept signals that pass ``_is_fresh``.  A stale FrameSignal
        from a track that has since stopped will still live in self._latest;
        returning it makes ``request_frame`` deliver an 8x8 placeholder
        because the underlying track is gone — the VLM then sees nothing.
        """
        ev = self._frame_events.setdefault(pid, asyncio.Event())
        t0 = asyncio.get_event_loop().time()
        deadline = t0 + timeout

        # TOCTOU: clear event, then re-check before blocking.
        ev.clear()
        sig = self._latest_signal(pid)
        if sig is not None and self._is_fresh(sig):
            logger.info(
                "camera frame pid={!r}  track={}  age_ms={:.0f}  (immediate)",
                pid, sig.track_id, (now_us() - sig.pts_us) / 1_000,
            )
            return sig

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                sig = self._latest_signal(pid)
                logger.warning(
                    "camera timeout pid={!r}  waited={:.1f}s  "
                    "latest_frame_age_ms={}  tracks_seen={}",
                    pid, timeout,
                    f"{(now_us() - sig.pts_us) / 1_000:.0f}" if sig else "none",
                    len([k for k in self._latest if k[0] == pid]),
                )
                return None
            try:
                await asyncio.wait_for(ev.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                logger.debug(
                    "still waiting for camera pid={!r}  elapsed={:.1f}s",
                    pid, asyncio.get_event_loop().time() - t0,
                )
                ev.clear()
                continue

            # Event fired — a new FrameSignal arrived.  Still require freshness
            # so we don't pick up a max-pts_us signal from a stopped track.
            sig = self._latest_signal(pid)
            if sig is not None and self._is_fresh(sig):
                logger.info(
                    "camera frame pid={!r}  track={}  age_ms={:.0f}  after {:.1f}s",
                    pid, sig.track_id, (now_us() - sig.pts_us) / 1_000,
                    asyncio.get_event_loop().time() - t0,
                )
                return sig
            ev.clear()

    def _is_fresh(self, sig: FrameSignal) -> bool:
        return now_us() - sig.pts_us < self._frame_max_age_us

    def _schedule_camera_off(self, pid: str) -> None:
        """Schedule stopCamera for ``pid`` after the grace period.

        Replaces any existing pending timer.  If a new query arrives before
        the timer fires, ``handle_query`` cancels it so the camera stays on.
        """
        old = self._camera_off_timers.pop(pid, None)
        if old and not old.done():
            old.cancel()

        async def _off():
            try:
                await asyncio.sleep(self._camera_grace_s)
                if pid not in self._camera_held:
                    # Claim before the await so no concurrent _ensure_camera_on
                    # can see True and skip sending startCamera after we stop.
                    self._camera_on[pid] = False
                    await self._client_control(pid, "stopCamera")
            except asyncio.CancelledError:
                pass

        self._camera_off_timers[pid] = asyncio.create_task(_off())

    # ── frame tracking ────────────────────────────────────────────────────────

    async def on_frame(self, sig: FrameSignal) -> None:
        prev = self._latest.get((sig.participant_id, sig.track_id))
        self._latest[(sig.participant_id, sig.track_id)] = sig
        # Log the very first frame per track so we can confirm signals arrive.
        if prev is None:
            logger.info(
                "first frame signal  pid={!r}  track={}  age_ms={:.0f}",
                sig.participant_id, sig.track_id,
                (now_us() - sig.pts_us) / 1_000,
            )
        # Wake any waiter in _wait_for_camera_frame.
        ev = self._frame_events.get(sig.participant_id)
        if ev is not None:
            ev.set()

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        # Use pts_us (real Unix timestamp) not seq (per-track counter).
        # seq restarts from 1 on each camera restart, so the old track's
        # stale entry wins max(seq) for hundreds of frames on the new track.
        return max(candidates, key=lambda s: s.pts_us)

    # ── participant lifecycle ─────────────────────────────────────────────────

    async def on_participant_left(self, pid: str) -> None:
        """Clear brain-owned per-pid state when a participant leaves."""
        for k in [k for k in self._latest if k[0] == pid]:
            del self._latest[k]
        self._frame_events.pop(pid, None)
        self._camera_on.pop(pid, None)
        self._camera_held.discard(pid)
        timer = self._camera_off_timers.pop(pid, None)
        if timer and not timer.done():
            timer.cancel()

    def shutdown(self) -> None:
        """Cancel any pending grace-period off timers (best-effort)."""
        for t in self._camera_off_timers.values():
            t.cancel()
