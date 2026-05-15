# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmAgent — vision Q&A driven by voice, text, or "ping".

Inputs
------
* Audio chunks (mic):  VAD detects an utterance, STT turns it into text,
                       which is then dispatched as a query.
* Data messages:       text payload is dispatched as a query directly.
* "ping" data message: literal text "ping" (case-insensitive) is replaced
                       with the configured default prompt before dispatch.

Each query is answered against the latest video frame for that participant
via a streaming VLM call.  The response goes back two ways:

* ``vlm.response`` data message — the assembled text reply.
* ``xr-hub-return-{pid}`` audio track — sentence-by-sentence Piper TTS,
  started in parallel as soon as each sentence completes.

Interruption
------------
A new query cancels any in-flight response for the same participant.  The
dispatcher cancels the running task, awaits cleanup, and unconditionally
calls ``flush_return_audio`` before starting the new one.

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
import re
import time

import httpx
import numpy as np
from loguru import logger
from xr_ai_agent import (AudioChunk, DataMessage, FrameSignal,
                          ParticipantEvent, ProcessorEndpoint)
from xr_ai_logging import print_task_done_banner

import pathlib

from audio import chunks_to_wav, now_us, rms, wav_to_chunks
from pixels import encode_image, frame_to_pil
from services import PoseClient, SttClient, TtsClient, VlmClient
from voice import VoiceState


DEFAULT_SYSTEM_PROMPT = (
    "You are an XR assistant speaking directly to the person wearing the headset. "
    "You can see their live camera feed and help them understand their environment.\n"
    "\n"
    "Style:\n"
    "- Speak directly to me in second person: 'You are looking at…', 'I can see…', "
    "'In front of you there is…'. Never refer to 'the user' in the third person.\n"
    "- Reply in plain conversational English — never JSON, code, or markdown.\n"
    "- Keep replies to 10-15 words by default. Only go longer when I "
    "explicitly ask for detail (e.g. 'describe in detail', 'tell me more', "
    "'elaborate', 'explain').\n"
    "- If I say 'stop', ask you to be quiet, or ask you to stop "
    "talking, just acknowledge briefly with something like 'Okay, I will stop.' "
    "and say nothing else."
)


class SimpleVlmAgent:

    def __init__(
        self,
        ep:  ProcessorEndpoint,
        stt: SttClient,
        vlm: VlmClient,
        tts: TtsClient,
        *,
        pose:               PoseClient | None = None,
        default_prompt:     str   = "Describe what you see.",
        system_prompt:      str   = DEFAULT_SYSTEM_PROMPT,
        silence_threshold:  float = 0.01,
        silence_duration:   float = 0.8,
        min_speech:         float = 0.3,
        frame_max_age_s:     float = 2.0,
        camera_on_timeout_s: float = 15.0,
        camera_grace_s:      float = 5.0,
        pose_hz:             float = 2.0,
        pose_max_age_s:      float = 0.6,
        pose_scratch_dir:    pathlib.Path = pathlib.Path("/dev/shm/xr-ai/pose-in"),
    ) -> None:
        self._ep   = ep
        self._stt  = stt
        self._vlm  = vlm
        self._tts  = tts
        self._pose = pose

        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_frame(self._on_frame)
        self._ep.on_participant(self._on_participant)

        self._default_prompt    = default_prompt
        self._system_prompt     = system_prompt
        self._vad_threshold     = silence_threshold
        self._vad_silence_s     = silence_duration
        self._vad_min_s         = min_speech
        self._frame_max_age_us  = int(frame_max_age_s * 1_000_000)
        self._camera_on_timeout = camera_on_timeout_s
        self._camera_grace_s    = camera_grace_s

        # Pose loop state — opportunistic, never triggers camera-on.
        self._pose_min_period_s = 1.0 / max(pose_hz, 0.1)    # throttle floor
        self._pose_max_age_s    = float(pose_max_age_s)      # drop if older
        self._pose_scratch_dir  = pose_scratch_dir
        self._pose_last_pts: dict[str, int] = {}             # pid → last sent ts
        # Event raised whenever a new FrameSignal arrives.  The pose loop
        # waits on this rather than sleeping on a fixed timer — guarantees
        # we always reach for the *latest* arrived frame, never a queued
        # one.  asyncio.Event coalesces multiple sets into one wakeup, so
        # bursts of incoming signals during a long inference don't queue.
        self._pose_event = asyncio.Event()
        if self._pose is not None:
            self._pose_scratch_dir.mkdir(parents=True, exist_ok=True)

        # IMU + camera-meta state pushed by the web client publisher.
        # `_imu[pid]` holds a rolling orientation (device ← world) integrated
        # from the gyro stream; we pass this to pose-mcp as an initial
        # guess so PnP converges faster + drops fewer frames.  Pure gyro
        # is sufficient — even uncorrected drift is well below the
        # inter-frame motion magnitude PnP needs to recover.
        self._imu: dict[str, dict] = {}
        self._camera_meta: dict[str, dict] = {}

        self._voice:  dict[str, VoiceState]              = {}
        self._latest: dict[tuple[str, str], FrameSignal] = {}

        # Camera on demand state
        self._camera_on: dict[str, bool]           = {}  # pid → agent requested camera on
        self._camera_held: set[str]                = set()  # pids in active query
        self._camera_off_timers: dict[str, asyncio.Task] = {}  # pid → delayed-off task
        self._frame_events: dict[str, asyncio.Event]     = {}  # pid → event set on new frame

    # ── audio path: VAD → STT → query ─────────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        pid = chunk.participant_id
        vs  = self._get_voice(pid, chunk.sample_rate, chunk.channels)

        chunk_s = chunk.samples / max(chunk.sample_rate, 1)
        if rms(chunk.data) >= self._vad_threshold:
            vs.chunks.append(chunk)
            vs.speech_s += chunk_s
            vs.silent_s  = 0.0
        else:
            if vs.chunks:
                vs.chunks.append(chunk)
            vs.silent_s += chunk_s

        # Speculative camera-on: the moment speech crosses min_speech, tell
        # the client to start the camera so it warms up in parallel with the
        # user finishing their sentence and STT processing.  By the time the
        # query dispatches the camera is usually already streaming.
        if (vs.speech_s >= self._vad_min_s
                and vs.speech_s - chunk_s < self._vad_min_s
                and not vs.transcribing):
            old_timer = self._camera_off_timers.pop(pid, None)
            if old_timer and not old_timer.done():
                old_timer.cancel()
            asyncio.create_task(self._ensure_camera_on(pid))

        if (vs.silent_s  >= self._vad_silence_s
                and vs.speech_s >= self._vad_min_s
                and not vs.transcribing):
            utterance   = vs.chunks[:]
            vs.chunks   = []
            vs.speech_s = vs.silent_s = 0.0
            vs.transcribing = True
            asyncio.create_task(self._handle_audio_utterance(pid, utterance, vs))
        elif (vs.silent_s >= self._vad_silence_s
                and vs.speech_s < self._vad_min_s
                and vs.chunks):
            vs.chunks   = []
            vs.speech_s = 0.0

    def _get_voice(self, pid: str, sample_rate: int = 16000, channels: int = 1) -> VoiceState:
        if pid not in self._voice:
            self._voice[pid] = VoiceState(sample_rate=sample_rate, channels=channels)
        return self._voice[pid]

    async def _handle_audio_utterance(
        self, pid: str, chunks: list[AudioChunk], vs: VoiceState,
    ) -> None:
        try:
            text = (await self._stt.transcribe(chunks_to_wav(chunks))).strip()
            if not text:
                return
            logger.info("audio query  pid={!r}  {!r}", pid, text[:80])
            await self._dispatch_query(pid, text, pts_us=now_us())
        except httpx.HTTPError as exc:
            logger.error("stt error pid={!r}: {}", pid, exc)
        finally:
            vs.transcribing = False

    # ── data path: text → query (with "ping" → default prompt) ────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        # IMU + camera-meta from the web client publisher.  These are
        # JSON-encoded best-effort messages on dedicated topics; intercept
        # them BEFORE the free-text dispatch path so they don't surface as
        # user "queries" against the VLM.
        if msg.topic == "imu":
            self._ingest_imu(msg.participant_id, msg.data)
            return
        if msg.topic == "camera_meta":
            self._ingest_camera_meta(msg.participant_id, msg.data)
            return
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        logger.info("data query  pid={!r}  {!r}", msg.participant_id, text[:80])
        await self._dispatch_query(msg.participant_id, text, pts_us=msg.pts_us)

    # ── IMU / camera meta ingest ──────────────────────────────────────────────

    def _ingest_imu(self, pid: str, payload: bytes) -> None:
        """Parse a batch from the web client's IMU publisher and integrate
        gyro into a rolling orientation prior for ``pid``.

        We keep things deliberately simple: pure gyro integration in the
        device frame.  Accelerometer is buffered for future tilt correction
        but not used in this pass — even uncorrected gyro is enough to give
        pose-mcp a small enough initial-guess error for PnP to converge
        faster.
        """
        try:
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            return
        samples_g = data.get("g") or []
        dt_ms     = float(data.get("dt", 16))
        if not samples_g:
            return
        st = self._imu.setdefault(pid, {
            "orient": np.eye(3, dtype=np.float64),   # device frame ← world
            "last_seen_us": 0,
            "samples_seen": 0,
        })
        dt_s = max(1e-4, dt_ms / 1000.0)
        for g in samples_g:
            wx, wy, wz = float(g[0]), float(g[1]), float(g[2])
            # First-order rotation matrix update: R_new = R · exp(skew(ω) dt).
            # For small dt this is well-approximated by R · (I + skew(ω) dt),
            # but using Rodrigues handles larger samples (e.g. 50 ms ones)
            # without numerical drift.
            theta = np.array([wx, wy, wz]) * dt_s
            angle = float(np.linalg.norm(theta))
            if angle < 1e-9:
                continue
            axis = theta / angle
            K = np.array([
                [0,        -axis[2],  axis[1]],
                [ axis[2],  0,       -axis[0]],
                [-axis[1],  axis[0],  0      ],
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            st["orient"] = st["orient"] @ dR
            st["samples_seen"] += 1
        st["last_seen_us"] = int(data.get("t", 0)) * 1000

    def _ingest_camera_meta(self, pid: str, payload: bytes) -> None:
        """Stash the camera metadata the client published once on startup
        and — when we can guess a sensible FOV from it — pin pose-mcp's
        intrinsics so it can skip the MoGe-based per-frame estimation
        (which is the biggest source of pose ambiguity on indoor flat
        scenes).

        Heuristic: width/height + facingMode → typical FOV bucket.  Errs
        on the conservative side (slightly under typical) — too-narrow
        FOV produces over-large 3D distances, which is less catastrophic
        than a too-wide FOV (which collapses points toward the camera).
        """
        try:
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            return
        self._camera_meta[pid] = data
        logger.info(
            "camera meta  pid={!r}  {}x{} fps={} facing={} label={!r}",
            pid, data.get("width"), data.get("height"),
            data.get("frame_rate"), data.get("facing"),
            (data.get("label") or "")[:50],
        )
        fov = _guess_fov_from_meta(data)
        if fov is not None and self._pose is not None:
            asyncio.create_task(self._push_fov_to_pose(fov))

    async def _push_fov_to_pose(self, fov_deg: float) -> None:
        assert self._pose is not None
        try:
            r = await self._pose.set_camera_fov(fov_deg)
            logger.info("pose-mcp FOV pinned to {:.1f}° (response={})", fov_deg, r)
        except Exception as exc:
            logger.warning("pose-mcp set_camera_fov failed: {}", exc)

    # ── interruptable query dispatch ──────────────────────────────────────────

    async def _dispatch_query(self, pid: str, text: str, *, pts_us: int) -> None:
        """Cancel any in-flight response for ``pid``, flush queued audio,
        then start the new query as a tracked task."""
        vs = self._get_voice(pid)

        async with vs.dispatch_lock:
            old = vs.current_task
            if old is not None and not old.done():
                logger.info("interrupt pid={!r} — cancelling in-flight response", pid)
                old.cancel()
                try:
                    await old
                except (asyncio.CancelledError, Exception):
                    pass

            await self._ep.flush_return_audio(pid)
            vs.current_task = asyncio.create_task(self._handle_query(pid, text, pts_us))

    async def _handle_query(self, pid: str, text: str, pts_us: int) -> None:
        query = self._default_prompt if text.lower().strip() == "ping" else text

        # Cancel any pending camera-off so a rapid follow-up query doesn't
        # see the camera turn off between the previous grace period firing.
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()

        self._camera_held.add(pid)
        t0 = time.monotonic()
        status = "done"
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
                    await self._say(pid, "Camera unavailable, please try again.", pts_us)
                    return

            frame = await self._ep.request_frame(sig)
            if frame is None:
                await self._say(pid, "Frame data unavailable — please retry.", pts_us)
                return

            image_url = encode_image(frame_to_pil(frame))
            logger.info(
                "vlm  pid={!r}  {}x{}  query={!r}",
                pid, frame.width, frame.height, query[:60],
            )

            await self._ep.set_status("processing", pid)
            try:
                full_response = await self._stream_and_speak(
                    pid, image_url, query, frame.pts_us,
                )
            finally:
                await self._ep.set_status("idle", pid)

            if full_response is not None:
                await self._reply(pid, full_response, frame.pts_us)
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            self._camera_held.discard(pid)
            # After the query, keep camera on for a grace period so rapid
            # follow-up queries skip the startup delay.  Then send stopCamera.
            self._schedule_camera_off(pid)
            print_task_done_banner(
                "simple-vlm-example",
                status=status,
                detail=f"pid={pid!r}  query={query[:60]!r}",
                duration_s=time.monotonic() - t0,
            )

    async def _stream_and_speak(
        self, pid: str, image_url: str, query: str, fallback_pts_us: int,
    ) -> str | None:
        """Run streaming VLM → sentence-batched TTS in parallel."""
        full_response = ""
        sentence_buf  = ""
        tts_queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
        pending_synth: list[asyncio.Task] = []

        async def _audio_sender() -> None:
            while True:
                task = await tts_queue.get()
                if task is None:
                    break
                try:
                    wav = await task
                    for chunk in wav_to_chunks(wav, pid):
                        await self._ep.send_return_audio(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "tts audio error pid={!r}: {}", pid, exc,
                    )

        sender = asyncio.create_task(_audio_sender())

        try:
            try:
                async for token in self._vlm.stream(
                    image_url, query, system_prompt=self._system_prompt,
                ):
                    full_response += token
                    sentence_buf  += token
                    while True:
                        m = re.search(r'(?<=[.!?])\s+', sentence_buf)
                        if not m:
                            break
                        sentence     = sentence_buf[:m.start() + 1].strip()
                        sentence_buf = sentence_buf[m.end():]
                        if sentence:
                            t = asyncio.create_task(self._tts.synthesize(sentence))
                            pending_synth.append(t)
                            await tts_queue.put(t)
                if sentence_buf.strip():
                    t = asyncio.create_task(self._tts.synthesize(sentence_buf.strip()))
                    pending_synth.append(t)
                    await tts_queue.put(t)
            except httpx.HTTPError as exc:
                logger.error("vlm-server error: {}", exc)
                await tts_queue.put(None)
                await sender
                await self._reply(pid, "VLM server unavailable — please retry.", fallback_pts_us)
                return None

            await tts_queue.put(None)
            await sender
            full_response = full_response.strip()
            logger.info("vlm response  pid={!r}  {} chars", pid, len(full_response))
            return full_response

        except asyncio.CancelledError:
            logger.info("response cancelled pid={!r}", pid)
            for t in pending_synth:
                t.cancel()
            sender.cancel()
            for t in (*pending_synth, sender):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            raise

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
            # (speculative _on_audio + _handle_query) can't both see False
            # and each send startCamera.
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
        the timer fires, ``_handle_query`` cancels it so the camera stays on.
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

    # ── reply helpers ─────────────────────────────────────────────────────────

    async def _reply(self, pid: str, text: str, pts_us: int) -> None:
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="vlm.response",
            pts_us=pts_us,
            data=text.encode(),
        ))

    async def _say(self, pid: str, text: str, pts_us: int) -> None:
        """Send a short canned reply on both data + audio channels (no VLM)."""
        await self._reply(pid, text, pts_us)
        try:
            wav = await self._tts.synthesize(text)
            for chunk in wav_to_chunks(wav, pid):
                await self._ep.send_return_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error(
                "tts error pid={!r}: {}", pid, exc,
            )

    # ── frame tracking ────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
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
        # Wake the pose loop so it always reaches for the freshest signal.
        # asyncio.Event coalesces multiple sets between processings, so a
        # burst of incoming frames doesn't queue — the loop just picks the
        # latest signal next time around.
        self._pose_event.set()

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        # Use pts_us (real Unix timestamp) not seq (per-track counter).
        # seq restarts from 1 on each camera restart, so the old track's
        # stale entry wins max(seq) for hundreds of frames on the new track.
        return max(candidates, key=lambda s: s.pts_us)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            return
        pid = event.participant_id
        vs  = self._voice.pop(pid, None)
        if vs is not None and vs.current_task is not None and not vs.current_task.done():
            vs.current_task.cancel()
        for k in [k for k in self._latest if k[0] == pid]:
            del self._latest[k]
        self._frame_events.pop(pid, None)
        self._camera_on.pop(pid, None)
        self._camera_held.discard(pid)
        timer = self._camera_off_timers.pop(pid, None)
        if timer and not timer.done():
            timer.cancel()

    # ── pose path: opportunistic estimate_pose at ~pose_hz ────────────────────

    async def _pose_loop(self) -> None:
        """Send the freshest available frame per pid to pose-mcp.  Wakes on
        each new ``FrameSignal`` arrival (not on a timer), then throttles
        to at most ``1 / pose_hz`` frames per second.

        Stale frames (older than ``pose_max_age_s`` from wall clock) are
        dropped so the viewer never falls behind the live stream — if
        the pipeline gets behind we skip frames rather than queue them.

        Never asks the client to turn the camera on — pose is a bystander
        to the VLM flow and only acts on frames already arriving.
        """
        logger.info(
            "pose loop running  min_period={:.2f}s  max_age={:.2f}s  scratch={}",
            self._pose_min_period_s, self._pose_max_age_s, self._pose_scratch_dir,
        )
        idle_logged = False
        try:
            while True:
                # Block until a new frame arrives.  Short timeout so the
                # "idle" log fires even when nothing is happening.
                try:
                    await asyncio.wait_for(self._pose_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                self._pose_event.clear()

                pids = {pid for pid, _ in self._latest}
                if not pids:
                    if not idle_logged:
                        logger.info("pose loop idle — no participants with frames yet")
                        idle_logged = True
                    continue
                idle_logged = False
                for pid in pids:
                    try:
                        await self._estimate_one(pid)
                    except Exception as exc:
                        logger.opt(exception=True).warning(
                            "pose iteration failed pid={!r}: {}", pid, exc,
                        )
                # Throttle floor: if pose-mcp returned faster than the
                # configured rate, give the camera time to produce a new
                # frame before looping.  Any signals that arrive during
                # this sleep coalesce into a single Event set, so we wake
                # up to the latest one rather than processing a queue.
                await asyncio.sleep(self._pose_min_period_s)
        except asyncio.CancelledError:
            raise

    async def _estimate_one(self, pid: str) -> None:
        assert self._pose is not None
        sig = self._latest_signal(pid)
        if sig is None:
            return
        # Wall-clock age check (separate from the VLM path's _is_fresh,
        # which allows up to 2 s).  Pose lag compounds visually in Rerun,
        # so we want sub-second freshness or we drop the frame outright —
        # never let the pipeline get behind by waiting on a stale frame.
        age_s = (now_us() - sig.pts_us) / 1_000_000.0
        if age_s > self._pose_max_age_s:
            logger.debug("pose: dropping stale frame pid={!r}  age={:.2f}s", pid, age_s)
            return
        if self._pose_last_pts.get(pid) == sig.pts_us:
            return   # we already sent this exact frame
        frame = await self._ep.request_frame(sig)
        if frame is None:
            return

        img = frame_to_pil(frame)
        out_path = self._pose_scratch_dir / f"{_safe_pid(pid)}.png"
        # Save to a sibling tmp + rename so pose-mcp can never read a half-
        # written PNG if the loop races itself on an unexpectedly slow GPU.
        tmp_path = out_path.with_suffix(".png.tmp")
        img.save(tmp_path, format="PNG")
        tmp_path.replace(out_path)

        # Snapshot the latest gyro-integrated orientation as a PnP prior.
        # `_ingest_imu` is keeping this fresh in the background — we just
        # read it.  No prior is sent for clients that don't publish IMU.
        prior_R = None
        st = self._imu.get(pid)
        if st is not None and st["samples_seen"] > 0:
            prior_R = list(float(x) for x in st["orient"].flatten())

        t0 = time.monotonic()
        try:
            result = await self._pose.estimate_pose(
                str(out_path), timestamp_us=frame.pts_us,
                prior_orientation=prior_R,
            )
        except Exception as exc:
            logger.warning("pose-mcp call failed pid={!r}: {}", pid, exc)
            return
        self._pose_last_pts[pid] = sig.pts_us
        dt_ms = (time.monotonic() - t0) * 1000.0

        if result.get("error"):
            logger.warning("pose-mcp error pid={!r}: {}", pid, result["error"])
            return

        # First successful result per pid is loud (so the operator sees the
        # pipeline came up); steady-state is one INFO per call so the logs
        # stay readable.  Drop pose_hz in the YAML if 2 Hz is too chatty.
        logger.info(
            "pose  pid={!r}  state={}  t={}  q={}  inliers={}  kfs={}  ({:.0f} ms)",
            pid, result.get("state"),
            _fmt3(result.get("translation_m")),
            _fmt4(result.get("quaternion")),
            result.get("num_inliers"), result.get("num_keyframes"), dt_ms,
        )

        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="pose.update",
            pts_us=frame.pts_us,
            data=json.dumps({
                "state":         result.get("state"),
                "translation_m": result.get("translation_m"),
                "quaternion":    result.get("quaternion"),
                "num_inliers":   result.get("num_inliers"),
                "num_keyframes": result.get("num_keyframes"),
                "ts_us":         result.get("ts_us"),
            }).encode(),
        ))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        pose_task: asyncio.Task | None = None
        if self._pose is not None:
            pose_task = asyncio.create_task(self._pose_loop(), name="pose-loop")
        try:
            await self._ep.run()
        finally:
            if pose_task is not None:
                pose_task.cancel()
                try:
                    await pose_task
                except (asyncio.CancelledError, Exception):
                    pass
            for t in self._camera_off_timers.values():
                t.cancel()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


def _safe_pid(pid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)


def _guess_fov_from_meta(meta: dict) -> float | None:
    """Coarse FOV-x guess from the published camera metadata.

    The web platform doesn't expose true intrinsics on `MediaTrackSettings`,
    so we triangulate from facingMode + resolution + the user-agent.
    Numbers are typical of consumer cameras as of 2026 — well-tuned
    enough to seed pose-mcp's K matrix and skip the (noisier) MoGe
    per-frame estimation, but the operator can always run a proper
    calibration and call ``pose-mcp.set_camera_fov`` directly to
    override.
    """
    facing = (meta.get("facing") or "").lower()
    ua     = (meta.get("user_agent") or "").lower()
    width  = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    # Rough device class — affects expected sensor + lens.
    if "iphone" in ua or "ipad" in ua:
        # iPhone main camera ≈ 26 mm equiv ⇒ ~67° H-FOV; front ≈ 75°.
        return 75.0 if facing == "user" else 67.0
    if "android" in ua and ("mobile" in ua or "phone" in ua):
        return 72.0 if facing == "user" else 68.0
    # Laptop webcams cluster around 55-65° H-FOV.
    return 60.0


def _fmt3(v):
    if v is None:
        return "—"
    return "[{:+.2f}, {:+.2f}, {:+.2f}]".format(*v)


def _fmt4(v):
    if v is None:
        return "—"
    return "[{:+.2f}, {:+.2f}, {:+.2f}, {:+.2f}]".format(*v)
