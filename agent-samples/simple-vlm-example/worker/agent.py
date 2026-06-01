# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmAgent — vision Q&A driven by voice, text, or "ping".

Inputs
------
* Audio chunks (mic):  VAD detects an utterance, STT turns it into text,
                       which is then dispatched as a query.  If any
                       ``magic_phrases`` are configured, the transcript
                       must begin with one of them (case-insensitive,
                       strict prefix — no leading filler words) or the
                       utterance is dropped.  Configure several phrases
                       to accept multiple wordings (e.g. "agent" and
                       "hey agent") without resorting to fuzzy matching.
                       This is the opt-in gate that keeps ambient
                       conversation from triggering the agent.  When a
                       phrase matches, the prefix is stripped before
                       dispatch and a short follow-up window opens — the
                       next utterance from that participant within
                       ``followup_grace_s`` seconds bypasses the gate so
                       a natural pause between the phrase and the
                       question still works.  ``stop`` and related
                       interrupt phrases always pass through and do not
                       extend the follow-up window.
* Data messages:       text payload is dispatched as a query directly.
                       The magic-phrase gate does not apply to this path.
* "ping" data message: literal text "ping" (case-insensitive) is replaced
                       with the configured default prompt before dispatch.
                       Note: a *spoken* "ping" is gated by the magic phrase
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
import dataclasses
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

from audio import int16_pcm_to_wav, now_us, wav_to_chunks
from pixels import encode_image, frame_to_pil
from voice import VoiceState

# Transcripts matching this pattern bypass the magic-phrase gate so the
# user can interrupt a response mid-flight without having to start with
# the configured phrase.
_STOP_RE = re.compile(
    r'^\s*(?:\S+\s+){0,2}'               # up to 2 optional filler words
    r'(?:stop(?:\s+\w+){0,2}|be\s+quiet|quiet|shut\s+up)'
    r'\s*[.!?]?\s*$',
    re.IGNORECASE,
)


def _build_chime_chunks(sample_rate: int) -> list[AudioChunk]:
    """Synthesize the listening-chime and pre-slice into AudioChunks at
    ``sample_rate``.

    Two-tone perfect-fifth ding (880 + 1320 Hz) with exponential decay,
    ~250 ms total, mono int16 PCM. The sample rate MUST match the rate
    of the rest of the return audio track (TTS) because LiveKit's
    AudioSource is locked to the first chunk's params and rejects later
    frames at a different rate (InvalidState).
    The `participant_id` is patched per-send in `_send_chime`.
    """
    dur  = 0.25
    t    = np.linspace(0.0, dur, int(sample_rate * dur), endpoint=False, dtype=np.float32)
    tone = 0.55 * np.sin(2 * np.pi * 880.0 * t) + 0.30 * np.sin(2 * np.pi * 1320.0 * t)
    env  = np.exp(-t * 8.0).astype(np.float32)
    pcm  = (tone * env * 0.5 * 32767.0).clip(-32768, 32767).astype(np.int16)
    wav  = int16_pcm_to_wav(pcm.tobytes(), sample_rate, channels=1)
    return wav_to_chunks(wav, participant_id="")


def _read_wav_sample_rate(wav_bytes: bytes) -> int:
    """Pull the sample rate from a WAV blob without decoding the audio."""
    import io, wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate()


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


class SimpleVlmAgent:

    def __init__(
        self,
        ep:  ProcessorEndpoint,
        stt: STTService,
        vlm: VLMService,
        tts: TTSService,
        *,
        default_prompt:     str   = "Describe what you see.",
        system_prompt:      str   = DEFAULT_SYSTEM_PROMPT,
        magic_phrases:      list[str] | tuple[str, ...] | str = (),
        listening_chime:    bool  = False,
        followup_grace_s:   float = 5.0,
        silence_duration:   float = 0.8,
        min_speech:         float = 0.3,
        silero_threshold:   float = 0.5,
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
        # Accept a single string OR a list of phrases. Normalize to a
        # tuple of lowercased, non-empty phrases. Defensive: empty YAML
        # values parse as None and would otherwise crash .strip().
        if isinstance(magic_phrases, str):
            _raw = [magic_phrases]
        else:
            _raw = list(magic_phrases) if magic_phrases else []
        self._magic_phrases: tuple[str, ...] = tuple(
            p.strip().lower() for p in _raw if p and p.strip()
        )
        # Pre-compile one regex covering every configured phrase.
        # Longest-first ordering picks the most specific match when one
        # phrase is a prefix of another (e.g., "agent" vs "agent buddy").
        # Inside each phrase, the literal space between words is treated
        # as "whitespace OR punctuation" so STT transcripts like
        # "Hey, agent." still match the configured "hey agent".
        self._magic_re: re.Pattern | None = None
        if self._magic_phrases:
            sep = r'[\s,.:;!?-]+'
            alts = "|".join(
                sep.join(re.escape(w) for w in p.split())
                for p in sorted(self._magic_phrases, key=len, reverse=True)
            )
            self._magic_re = re.compile(
                rf'^\s*(?:{alts})\b[\s,.:;!?-]*', re.IGNORECASE,
            )
        # Chime is built lazily at the TTS sample rate the first time
        # any TTS WAV passes through (greeting or VLM reply). That
        # guarantees the chime matches the rate the LiveKit AudioSource
        # is locked to and avoids probing TTS with a dummy synthesize
        # call (Piper crashes on whitespace input). See
        # _maybe_build_chime. Disabled without a configured phrase.
        self._listening_chime_enabled = listening_chime and bool(self._magic_phrases)
        self._chime_chunks: list[AudioChunk] | None = None
        # Follow-up grace: after a successful magic-phrase match, the
        # next utterance from the same pid within this window bypasses
        # the gate. Lets users say "hey agent" → pause → "what am I
        # looking at?" naturally instead of mashing the phrase onto the
        # front of every question. The window resets each time an
        # utterance is accepted so a conversation keeps flowing.
        self._followup_grace_s      = followup_grace_s
        self._followup_until: dict[str, float] = {}
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

    # ── audio path: VAD → STT → query ─────────────────────────────────────────

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
            # Gate priority:
            #   0. STOP — always wins. Cancels in-flight + canned ack,
            #      regardless of magic-phrase or followup state. Matched
            #      on both the raw transcript ("stop") AND on the
            #      magic-phrase-stripped tail ("hey agent, stop") so the
            #      fast path triggers either way.
            #   1. strict-prefix magic phrase match (new conversation)
            #   2. follow-up grace window still open (continuation)
            now_mono       = time.monotonic()
            in_followup    = self._followup_until.get(pid, 0.0) > now_mono
            stripped       = self._strip_magic_phrase(text)
            matched_magic  = stripped is not None
            stop_candidate = stripped if (matched_magic and stripped) else text
            if _STOP_RE.match(stop_candidate):
                logger.info("stop bypass pid={!r}  {!r}", pid, stop_candidate[:80])
                self._followup_until.pop(pid, None)
                await self._handle_stop(pid)
                self._schedule_camera_off(pid)
                return
            if matched_magic:
                query = stripped
            elif in_followup:
                query = text
                logger.info("followup query pid={!r} {!r}", pid, text[:80])
            else:
                logger.info("magic phrase missing pid={!r} text={!r}", pid, text[:80])
                self._followup_until.pop(pid, None)
                self._schedule_camera_off(pid)
                return
            # Chime on a fresh magic-phrase match only — follow-ups are
            # already inside the conversation and STOP is a different signal.
            if matched_magic and self._listening_chime_enabled:
                asyncio.create_task(self._send_chime(pid))
            if not query:
                # Magic phrase with no follow-up payload yet — open the
                # window so the user's next utterance counts as the
                # actual query without needing the phrase again.
                self._followup_until[pid] = now_mono + self._followup_grace_s
                logger.info(
                    "magic phrase only pid={!r} — awaiting followup ({:.1f}s)",
                    pid, self._followup_grace_s,
                )
                self._schedule_camera_off(pid)
                return
            # Real query about to be dispatched. Close the window — the
            # next utterance must re-introduce a magic phrase. Keeps
            # ambient speech after the answer from re-entering.
            self._followup_until.pop(pid, None)
            logger.info("audio query  pid={!r}  {!r}", pid, query[:80])
            await self._dispatch_query(pid, query, pts_us=now_us())
        except httpx.HTTPError as exc:
            logger.error("stt error pid={!r}: {}", pid, exc)
        finally:
            vs.transcribing = False

    def _strip_magic_phrase(self, text: str) -> str | None:
        """Gate STT output on the configured magic phrase(s).

        Strict-prefix match (case-insensitive): the transcript must begin
        with one of the configured phrases as a whole word — no leading
        filler words and no mid-sentence matches. Configure multiple
        phrases (e.g. "agent", "hey agent") to accept several wordings
        without falling back to fuzzy matching.

        Returns the remainder of the transcript with the matched phrase
        and any adjacent punctuation stripped, or ``None`` if no phrase
        is the strict prefix. With no phrases configured the gate is
        disabled and ``text`` is returned unchanged.
        """
        if self._magic_re is None:
            return text
        m = self._magic_re.match(text)
        return None if m is None else text[m.end():]

    # ── data path: text → query (with "ping" → default prompt) ────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        logger.info("data query  pid={!r}  {!r}", msg.participant_id, text[:80])
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
                    self._maybe_build_chime(wav)
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
            self._maybe_build_chime(wav)
            for chunk in wav_to_chunks(wav, pid):
                await self._ep.send_return_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error(
                "tts error pid={!r}: {}", pid, exc,
            )

    def _maybe_build_chime(self, wav_bytes: bytes) -> None:
        """Build the chime at the TTS sample rate the first time we see
        a real TTS WAV. No-op once built or when chime is disabled."""
        if self._chime_chunks is not None or not self._listening_chime_enabled:
            return
        try:
            sr = _read_wav_sample_rate(wav_bytes)
            self._chime_chunks = _build_chime_chunks(sr)
            logger.info("listening chime ready (sr={} Hz)", sr)
        except Exception as exc:
            logger.opt(exception=True).warning(
                "listening chime disabled (bad TTS wav header): {}", exc,
            )
            self._listening_chime_enabled = False

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
        await self._say(pid, "Okay, I will stop.", now_us())

    async def _send_chime(self, pid: str) -> None:
        """Emit the listening-chime on the return audio track. No-op
        when chime is disabled."""
        if self._chime_chunks is None:
            return
        pts0 = now_us()
        try:
            for i, ch in enumerate(self._chime_chunks):
                chunk = dataclasses.replace(
                    ch, participant_id=pid, pts_us=pts0 + i * 20_000,
                )
                await self._ep.send_return_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error(
                "chime send error pid={!r}: {}", pid, exc,
            )

    async def _greet(self, pid: str) -> None:
        """Speak a one-shot connection greeting that tells the user how to
        address the agent given the current magic-phrase setting."""
        phrases = list(self._magic_phrases)
        if not phrases:
            text = "Hi, I'm listening. Ask me anything about what you see."
        else:
            if len(phrases) == 1:
                phrase_list = f'"{phrases[0]}"'
            elif len(phrases) == 2:
                phrase_list = f'"{phrases[0]}" or "{phrases[1]}"'
            else:
                phrase_list = (
                    ", ".join(f'"{p}"' for p in phrases[:-1])
                    + f', or "{phrases[-1]}"'
                )
            text = (
                f"Hi, I'm listening. To talk to me, start your question "
                f"with {phrase_list}. For example, "
                f"\"{phrases[0]}, what am I looking at?\""
            )
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
            asyncio.create_task(self._greet(event.participant_id))
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
        self._followup_until.pop(pid, None)
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
