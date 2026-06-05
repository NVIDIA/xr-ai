# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmBrain — vision Q&A reasoning for the unified pipecat voice pipeline.

The brain implements :class:`xr_ai_pipecat.BrainProcessor`. Voice path is
handled upstream (VAD/STT → voice gate → ``GatedQueryFrame``); TTS is
handled downstream (``StreamingTtsProcessor``). This class owns ONLY:

* Camera-on-demand: ``startCamera`` / ``stopCamera`` on the
  ``clientControl`` topic, frame freshness checks, the grace timer that
  keeps the camera on across rapid follow-ups, and the
  ``on_user_started_speaking`` hook that speculatively warms up the
  camera the moment a user starts talking.
* Frame tracking: per-pid ``FrameSignal`` cache + wake event.
* The VLM streaming call: encode the latest frame and yield response
  tokens for downstream sentence-batched TTS.
* The data-channel side path: ``"ping"`` shortcut + ad-hoc text queries
  (the voice gate does not see these).

Hub ``DataMessage`` / ``FrameSignal`` events are not surfaced as
pipecat frames, so the brain registers callbacks on the transport's
``ProcessorEndpoint`` directly. Participant leave IS surfaced as a
``ParticipantLeftFrame`` (the transport bridges it), so teardown rides
the base ``BrainProcessor`` frame path rather than an endpoint
callback. The unified audio path goes through pipecat; this side path
stays on the IPC API.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import httpx
from loguru import logger
from pipecat.frames.frames import InterruptionFrame
from xr_ai_agent import DataMessage, FrameSignal
from xr_ai_logging import print_task_done_banner
from xr_ai_models import VLMService
from xr_ai_pipecat import BrainProcessor, GatedQueryFrame
from xr_ai_pipecat.transport import XRMediaHubTransport

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


def _now_us() -> int:
    return time.time_ns() // 1_000


class SimpleVlmBrain(BrainProcessor):
    """Camera + VLM brain for the simple-vlm-example sample.

    Voice path: voice gate → ``GatedQueryFrame`` → :meth:`handle_query`
    yields tokens, downstream TTS turns those into audio.

    Data-channel path: hub ``on_data`` builds a ``GatedQueryFrame``
    directly and re-uses the base brain's per-pid task tracking via
    :meth:`_spawn_query`.
    """

    def __init__(
        self,
        *,
        transport: XRMediaHubTransport,
        vlm: VLMService,
        default_prompt:      str   = "Describe what you see.",
        system_prompt:       str   = DEFAULT_SYSTEM_PROMPT,
        frame_max_age_s:     float = 2.0,
        camera_on_timeout_s: float = 15.0,
        camera_grace_s:      float = 5.0,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._vlm = vlm

        self._default_prompt = default_prompt
        self._system_prompt  = system_prompt

        self._frame_max_age_us  = int(frame_max_age_s * 1_000_000)
        self._camera_on_timeout = camera_on_timeout_s
        self._camera_grace_s    = camera_grace_s

        # Latest FrameSignal per (pid, track_id) — VLM consumes the
        # newest signal that passes _is_fresh.
        self._latest: dict[tuple[str, str], FrameSignal] = {}

        # Camera-on-demand state
        self._camera_on:        dict[str, bool]            = {}
        self._camera_held:      set[str]                   = set()
        self._camera_off_timers: dict[str, asyncio.Task]   = {}
        self._frame_events:     dict[str, asyncio.Event]   = {}

        # Side-path: register hub callbacks directly. Voice goes through
        # the pipecat pipeline; data + frame events do not yet have frame
        # equivalents.
        # Participant leave teardown is NOT registered here: the
        # transport bridges the hub ``ParticipantEvent(joined=False)`` to
        # a pipecat ``ParticipantLeftFrame``, and the base
        # ``BrainProcessor`` runs ``on_participant_left`` + ``_cancel_pid``
        # off that frame. Registering an endpoint callback too would tear
        # the same pid down twice on a single leave.
        ep = transport.endpoint
        ep.on_data(self._on_data)
        ep.on_frame(self._on_frame)

    # ── BrainProcessor overrides ──────────────────────────────────────────────

    async def handle_query(
        self, pid: str, text: str, fresh_match: bool,
    ) -> AsyncIterator[str]:
        """Drive one query end-to-end. Yields VLM tokens for downstream
        TTS. ``fresh_match`` is informational — both speech and data
        paths drive the same VLM call.

        Returns (not ``yield``s) the async iterator: the base
        ``BrainProcessor._run_query`` does ``result = await
        handle_query(...)`` then iterates ``result``, so this must be a
        coroutine that *returns* an ``AsyncIterator`` — not an async
        generator. Rewriting the body as ``async for ... yield`` would
        make this an async-gen function, and ``await``-ing an async-gen
        object raises; keep the ``return`` to honor the await-then-iterate
        contract.
        """
        return self._stream_query(pid, text)

    async def on_query_superseded(self, pid: str) -> None:
        """Interrupt the previous response's audio when a new query lands.

        Cancelling the prior brain task only stops *new* TextFrames from
        this processor. Without an explicit drain signal, the streaming
        TTS sender queue, the hub return-audio pacing pipe, and the
        jitter buffer keep delivering the previous answer — the user
        hears the old response finish before the new one starts. Push
        an ``InterruptionFrame`` so those layers flush at the source.

        The library default for this hook is a no-op (queue behind);
        this sample opts in to interrupt-on-supersede because vision
        Q&A turns are short and the user expects the new answer to
        cut in immediately.
        """
        await self.push_frame(InterruptionFrame())

    async def on_user_started_speaking(self, pid: str) -> None:
        """Speculative camera warmup at the leading edge of speech.

        Fires before the previous in-flight query is cancelled, so by
        the time the next utterance reaches the VLM the camera is
        usually already streaming. The previous PR-1-era hook ran on the
        VAD ``min_speech`` boundary; pipecat fires it earlier, on the
        same ``UserStartedSpeakingFrame`` that cancels in-flight work."""
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()
        try:
            await self._ensure_camera_on(pid)
        except Exception:
            logger.exception("camera warmup failed pid={!r}", pid)

    async def on_participant_left(self, pid: str) -> None:
        """Tear down per-pid state. Base class cancels in-flight tasks
        — we only own the camera + frame state."""
        self._latest = {k: v for k, v in self._latest.items() if k[0] != pid}
        self._frame_events.pop(pid, None)
        self._camera_on.pop(pid, None)
        self._camera_held.discard(pid)
        timer = self._camera_off_timers.pop(pid, None)
        if timer and not timer.done():
            timer.cancel()

    # ── data-channel side path ────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        logger.info("data query  pid={!r}  {!r}", msg.participant_id, text[:80])
        # Reuse the base class's per-pid in-flight tracking. "ping" is
        # an explicit user action — fresh_match=True mirrors the voice
        # path's fresh-magic-phrase semantics. The query payload swap
        # happens once here so handle_query stays paths-agnostic.
        query = self._default_prompt if text.lower() == "ping" else text
        await self._spawn_query(GatedQueryFrame(
            participant_id = msg.participant_id,
            text           = query,
            fresh_match    = True,
            pts_us         = msg.pts_us,
        ))

    # ── frame tracking ────────────────────────────────────────────────────────

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
        # pts_us is a real wall-clock timestamp; seq restarts on each
        # camera restart so it would pick a stale track's last entry.
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.pts_us)

    def _is_fresh(self, sig: FrameSignal) -> bool:
        return _now_us() - sig.pts_us < self._frame_max_age_us

    async def _wait_for_camera_frame(
        self, pid: str, timeout: float,
    ) -> FrameSignal | None:
        """Wait up to ``timeout`` for a fresh ``FrameSignal`` for ``pid``.

        Only signals that pass ``_is_fresh`` are accepted — a stale
        signal from a stopped track is still in self._latest, and
        returning it makes ``request_frame`` deliver an 8x8 placeholder
        because the underlying track is gone.
        """
        ev = self._frame_events.setdefault(pid, asyncio.Event())
        t0 = asyncio.get_event_loop().time()
        deadline = t0 + timeout

        ev.clear()
        sig = self._latest_signal(pid)
        if sig is not None and self._is_fresh(sig):
            logger.info(
                "camera frame pid={!r}  track={}  age_ms={:.0f}  (immediate)",
                pid, sig.track_id, (_now_us() - sig.pts_us) / 1_000,
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
                    f"{(_now_us() - sig.pts_us) / 1_000:.0f}" if sig else "none",
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

            sig = self._latest_signal(pid)
            if sig is not None and self._is_fresh(sig):
                logger.info(
                    "camera frame pid={!r}  track={}  age_ms={:.0f}  after {:.1f}s",
                    pid, sig.track_id, (_now_us() - sig.pts_us) / 1_000,
                    asyncio.get_event_loop().time() - t0,
                )
                return sig
            ev.clear()

    # ── camera on demand ──────────────────────────────────────────────────────

    async def _client_control(self, pid: str, action: str) -> None:
        await self._transport.send_return_data(DataMessage(
            participant_id = pid,
            topic          = "clientControl",
            pts_us         = _now_us(),
            data           = json.dumps({"action": action}).encode(),
        ))

    async def _ensure_camera_on(self, pid: str) -> None:
        """Send startCamera if we haven't already (idempotent).

        Claims the flag before the first await so concurrent callers
        (speculative on_user_started_speaking + handle_query) can't
        both see False and double-send.
        """
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
        """Schedule stopCamera for ``pid`` after the grace period.

        Replaces any pending timer. A new query arriving inside the
        grace window cancels this so the camera stays on.
        """
        old = self._camera_off_timers.pop(pid, None)
        if old and not old.done():
            old.cancel()

        async def _off():
            try:
                await asyncio.sleep(self._camera_grace_s)
                if pid not in self._camera_held:
                    self._camera_on[pid] = False
                    await self._client_control(pid, "stopCamera")
            except asyncio.CancelledError:
                # Expected: a newer query cancels this grace-period timer
                # before it fires (see the old.cancel() above). Nothing to do.
                pass

        self._camera_off_timers[pid] = asyncio.create_task(_off())

    # ── VLM streaming call ────────────────────────────────────────────────────

    async def _stream_query(self, pid: str, query: str) -> AsyncIterator[str]:
        """Acquire a fresh frame, encode it, stream VLM tokens.

        On any failure that the user should hear, yields a single canned
        string and returns — downstream TTS turns it into audio just
        like the streaming path.
        """
        # Cancel any pending camera-off so a rapid follow-up keeps the
        # camera on.
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()

        self._camera_held.add(pid)
        t0     = time.monotonic()
        status = "done"
        try:
            sig = self._latest_signal(pid)
            if not (sig and self._is_fresh(sig)):
                await self._ensure_camera_on(pid)
                sig = await self._wait_for_camera_frame(
                    pid, self._camera_on_timeout,
                )
                if sig is None:
                    self._camera_on[pid] = False
                    yield "Camera unavailable, please try again."
                    return

            frame = await self._transport.endpoint.request_frame(sig)
            if frame is None:
                yield "Frame data unavailable — please retry."
                return

            image_url = encode_image(frame_to_pil(frame))
            logger.info(
                "vlm  pid={!r}  {}x{}  query={!r}",
                pid, frame.width, frame.height, query[:60],
            )

            await self._transport.endpoint.set_status("processing", pid)
            try:
                try:
                    async for token in self._vlm.stream(
                        image_url, query, system_prompt=self._system_prompt,
                    ):
                        yield token
                except httpx.HTTPError as exc:
                    logger.error("vlm-server error: {}", exc)
                    yield "VLM server unavailable — please retry."
                    return
            finally:
                await self._transport.endpoint.set_status("idle", pid)
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            self._camera_held.discard(pid)
            # Keep the camera on for the grace window so rapid follow-up
            # queries don't pay the camera startup cost again.
            self._schedule_camera_off(pid)
            print_task_done_banner(
                "simple-vlm-example",
                status=status,
                detail=f"pid={pid!r}  query={query[:60]!r}",
                duration_s=time.monotonic() - t0,
            )
