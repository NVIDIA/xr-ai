# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmAgent — vision Q&A driven by voice, text, or "ping".

Inputs
------
* Audio chunks (mic):  VAD detects an utterance, STT turns it into text,
                       which is then handed to ``xr-ai-voicegate``. The
                       gate owns the magic-phrase, STOP, and follow-up
                       ladder — see ``utils/xr-ai-voicegate``. This
                       worker only wires handlers (``on_query``,
                       ``on_stop``, ``on_phrase_only``, ``on_drop``,
                       ``on_participant_joined``) and feeds transcripts.
* Data messages:       text payload is dispatched as a query directly.
                       The voice gate does not apply to this path.
* "ping" data message: literal text "ping" (case-insensitive) is replaced
                       with the configured default prompt before dispatch.
                       Note: a *spoken* "ping" is gated by the voice gate
                       like any other utterance; only the data-channel
                       shortcut is unaffected.

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
from loguru import logger
import numpy as np
from xr_ai_agent import (AudioChunk, DataMessage, FrameSignal,
                          ParticipantEvent, ProcessorEndpoint)
from xr_ai_logging import print_task_done_banner
from xr_ai_models import STTService, TTSService, VLMService
from xr_ai_vad import VadDetector
from xr_ai_voicegate import VoiceGate, VoiceGateConfig

from audio import int16_pcm_to_wav, now_us, wav_to_chunks
from pixels import encode_image, frame_to_pil
from voice import VoiceState


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


class _EpSink:
    """Adapter implementing ``xr_ai_voicegate.AudioSink`` over the hub's
    return-audio fan-out: explode the WAV into 20 ms AudioChunks via the
    worker's ``wav_to_chunks`` and stream them through ``send_return_audio``."""

    def __init__(self, ep: ProcessorEndpoint) -> None:
        self._ep = ep

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        for chunk in wav_to_chunks(wav_bytes, pid):
            await self._ep.send_return_audio(chunk)


class SimpleVlmAgent:

    def __init__(
        self,
        ep:  ProcessorEndpoint,
        stt: STTService,
        vlm: VLMService,
        tts: TTSService,
        *,
        voice_gate_cfg:      VoiceGateConfig,
        default_prompt:      str   = "Describe what you see.",
        system_prompt:       str   = DEFAULT_SYSTEM_PROMPT,
        silence_duration:    float = 0.8,
        min_speech:          float = 0.3,
        silero_threshold:    float = 0.5,
        frame_max_age_s:     float = 2.0,
        camera_on_timeout_s: float = 15.0,
        camera_grace_s:      float = 5.0,
    ) -> None:
        self._ep  = ep
        self._stt = stt
        self._vlm = vlm
        self._tts = tts

        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_frame(self._on_frame)
        self._ep.on_participant(self._on_participant)

        self._default_prompt    = default_prompt
        self._system_prompt     = system_prompt
        self._vad_silence_s         = silence_duration
        self._vad_min_s             = min_speech
        self._vad_silero_threshold  = silero_threshold
        self._frame_max_age_us  = int(frame_max_age_s * 1_000_000)
        self._camera_on_timeout = camera_on_timeout_s
        self._camera_grace_s    = camera_grace_s

        self._voice:  dict[str, VoiceState]              = {}
        self._latest: dict[tuple[str, str], FrameSignal] = {}

        # Camera on demand state
        self._camera_on: dict[str, bool]           = {}  # pid → agent requested camera on
        self._camera_held: set[str]                = set()  # pids in active query
        self._camera_off_timers: dict[str, asyncio.Task] = {}  # pid → delayed-off task
        self._frame_events: dict[str, asyncio.Event]     = {}  # pid → event set on new frame

        # Speech gating + greeting are owned by the voice gate.
        self._gate = VoiceGate(voice_gate_cfg, audio_sink=_EpSink(ep), tts=tts)
        self._gate.on_query(self._dispatch_from_voice)
        self._gate.on_stop(self._handle_stop)
        self._gate.on_phrase_only(self._on_phrase_only)
        self._gate.on_drop(self._on_drop)
        self._gate.on_participant_joined(self._greet)

    # ── audio path: VAD → STT → gate ──────────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        vs = self._get_voice(chunk.participant_id)
        assert vs.vad is not None
        # Hub delivers float32 LE PCM; VadDetector takes int16 LE PCM.
        f32  = np.frombuffer(chunk.data, dtype=np.float32)
        i16  = (np.clip(f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        await vs.vad.feed(i16, chunk.sample_rate)

    def _get_voice(self, pid: str) -> VoiceState:
        vs = self._voice.get(pid)
        if vs is None:
            vs = VoiceState()
            vs.vad = VadDetector(
                on_utterance      = lambda audio, sr, _pid=pid: self._on_vad_utterance(_pid, audio, sr),
                on_speech_start   = lambda _pid=pid: self._on_vad_speech_start(_pid),
                silence_duration  = self._vad_silence_s,
                min_speech        = self._vad_min_s,
                silero_threshold  = self._vad_silero_threshold,
            )
            self._voice[pid] = vs
        return vs

    async def _on_vad_speech_start(self, pid: str) -> None:
        """Speculative camera warmup the moment speech crosses min_speech.

        Fires while the user is still talking — by the time STT finishes,
        the camera is usually already streaming.  Skipped if a transcription
        for this pid is already in flight (the prior camera-on still applies).
        """
        vs = self._voice.get(pid)
        if vs is None or vs.transcribing:
            return
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()
        await self._ensure_camera_on(pid)

    async def _on_vad_utterance(self, pid: str, audio_bytes: bytes, sample_rate: int) -> None:
        vs = self._voice.get(pid)
        if vs is None or vs.transcribing:
            return
        vs.transcribing = True
        try:
            wav = int16_pcm_to_wav(audio_bytes, sample_rate)
            text = (await self._stt.transcribe(wav)).strip()
            if not text:
                return
            await self._gate.feed(pid, text)
        except httpx.HTTPError as exc:
            logger.error("stt error pid={!r}: {}", pid, exc)
        finally:
            vs.transcribing = False

    # ── voice-gate handlers ───────────────────────────────────────────────────

    async def _dispatch_from_voice(self, pid: str, query: str, fresh_match: bool) -> None:
        """Voice-gate ``on_query`` handler — chime + dispatch.

        Only chimes on a *fresh* magic-phrase match. Follow-up window
        continuations (``fresh_match=False``) skip the chime; the original
        pre-extraction code only chimed on case 2 and case 4, never on
        case 3, and this matches that behavior."""
        if fresh_match:
            asyncio.create_task(self._gate.play_chime(pid))
        await self._dispatch_query(pid, query, pts_us=now_us())

    async def _on_phrase_only(self, pid: str) -> None:
        """Voice-gate ``on_phrase_only`` handler — chime the
        acknowledgement so the user knows the wake word was heard while
        the follow-up window is open, then schedule a camera-off in case
        the user never follows up. ``_handle_query`` (case 2/3) cancels
        a pending off-timer when the next query lands."""
        await self._gate.play_chime(pid)
        self._schedule_camera_off(pid)

    async def _on_drop(self, pid: str, text: str) -> None:
        logger.info("voice gate dropped pid={!r} (text suppressed)", pid)
        logger.debug("voice gate dropped pid={!r} text={!r}", pid, text[:80])
        self._schedule_camera_off(pid)

    # ── data path: text → query (with "ping" → default prompt) ────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        logger.info("data query  pid={!r} (text suppressed)", msg.participant_id)
        logger.debug("data query  pid={!r}  {!r}", msg.participant_id, text[:80])
        await self._dispatch_query(msg.participant_id, text, pts_us=msg.pts_us)

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
                    self._gate.observe_tts_wav(wav)
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
            self._gate.observe_tts_wav(wav)
            for chunk in wav_to_chunks(wav, pid):
                await self._ep.send_return_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error(
                "tts error pid={!r}: {}", pid, exc,
            )

    async def _handle_stop(self, pid: str) -> None:
        """Cancel any in-flight response for this participant and play a
        canned ack. Bypasses the VLM/camera pipeline so a single 'stop'
        is acted on immediately."""
        vs = self._voice.get(pid)
        if vs is not None:
            async with vs.dispatch_lock:
                old = vs.current_task
                if old is not None and not old.done():
                    old.cancel()
                    try:
                        await old
                    except asyncio.CancelledError:
                        # Expected — that's the cancel we just issued.
                        pass
                    except Exception as exc:
                        # The old task may have failed in flight; we
                        # wanted it gone either way, so it's not
                        # actionable here, but log so it isn't silent.
                        logger.opt(exception=True).warning(
                            "in-flight task error during stop pid={!r}: {}",
                            pid, exc,
                        )
                await self._ep.flush_return_audio(pid)
                vs.current_task = None
        # Preserve the visible reply on ``vlm.response`` (clients listen
        # on it for the assistant's text) while letting the gate handle
        # the canned TTS + chime sample-rate observation.
        self._schedule_camera_off(pid)
        await self._reply(pid, "Okay, I will stop.", now_us())
        await self._gate.say_stop_ack(pid)

    async def _greet(self, pid: str) -> None:
        """Speak a one-shot connection greeting that tells the user how to
        address the agent given the current voice-gate setting."""
        help_text = self._gate.format_phrase_help()
        if help_text is None:
            text = "Hi, I'm listening. Ask me anything about what you see."
        else:
            text = f"Hi, I'm listening. {help_text}"
        try:
            await self._say(pid, text, now_us())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error(
                "greet error pid={!r}: {}", pid, exc,
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
            # Greet the user so they know the agent is listening and, if
            # a magic phrase is configured, how to address it. The speech
            # path is gated by default now, so without this hint a user
            # can easily think the agent is broken when it ignores them.
            asyncio.create_task(self._gate.participant_joined(event.participant_id))
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
        self._gate.forget(pid)
        timer = self._camera_off_timers.pop(pid, None)
        if timer and not timer.done():
            timer.cancel()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            await self._ep.run()
        finally:
            # Cancel any pending grace-period off timers.
            for t in self._camera_off_timers.values():
                t.cancel()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()
