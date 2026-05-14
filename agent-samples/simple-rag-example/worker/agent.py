# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleRagAgent — vision Q&A driven by voice or text, with optional doc grounding.

Inputs
------
* Audio chunks (mic):  VAD detects an utterance, STT turns it into text,
                       which is then dispatched as a query.
* Data messages:       text payload is dispatched as a query directly.
* "ping" data message: literal text "ping" (case-insensitive) is replaced
                       with the configured default prompt before dispatch.

Each query is answered by:
  1. Always calling rag-mcp ``retrieve`` (dense vector search via embedding-server)
     to fetch the top-k document chunks most semantically similar to the query.
  2. In parallel, requesting the latest camera frame (camera-on-demand).
  3. Sending the VLM a single multimodal turn: system prompt + retrieved
     chunks (when any) + image + question.  When retrieval returns nothing,
     the agent sends the question with the image alone — no fabricated context.
  4. Streaming the VLM reply as sentence-batched Piper TTS audio.

RAG context is optional — the agent works fine without doc-relevant queries.
When retrieval finds nothing, the agent behaves exactly like simple-vlm-example.

Text-only fallback
------------------
When no camera frame arrives within ``camera_on_timeout_s`` (e.g. the client
has no camera or hasn't granted access), the agent does not abort — it falls
through to a text-only VLM call and answers from the retrieved context (if
any) or from general knowledge.

Reply topics
------------
``vlm.response``        — assembled UTF-8 text (data channel).
``xr-hub-return-{pid}`` — sentence-by-sentence TTS audio (return track).

Camera on demand
----------------
Follows the same startCamera / stopCamera pattern as simple-vlm-example.
Speculative camera-on fires as soon as min_speech worth of audio has been
detected, so the frame is usually ready by the time STT finishes.

Interruption
------------
A new query cancels any in-flight response for the same participant, flushes
queued audio, then starts fresh.
"""
from __future__ import annotations

import asyncio
import json
import re
import time

import httpx
from loguru import logger
from xr_ai_agent import (AudioChunk, DataMessage, FrameSignal,
                          ParticipantEvent, ProcessorEndpoint)
from xr_ai_logging import print_task_done_banner

from audio import chunks_to_wav, now_us, rms, wav_to_chunks
from pixels import encode_image, frame_to_pil
from services import RagClient, SttClient, TtsClient, VlmClient
from voice import VoiceState


DEFAULT_SYSTEM_PROMPT = (
    "You are an XR vision assistant speaking directly to the person wearing "
    "the headset.  You can see their live camera feed.\n"
    "\n"
    "When reference text is provided in the user turn, ground your answer in "
    "that text.  When no reference text is provided, answer from what you see "
    "in the camera and your general knowledge.  Never fabricate document "
    "content that was not included in the reference text.\n"
    "\n"
    "Style:\n"
    "- Speak directly in second person: 'You are looking at…', 'I can see…'.\n"
    "- Reply in plain conversational English — never JSON, code, or markdown.\n"
    "- Keep replies to 10-15 words by default.  Only go longer when the user "
    "explicitly asks for detail (e.g. 'describe in detail', 'tell me more').\n"
    "- If the user says 'stop' or asks you to be quiet, acknowledge briefly "
    "and say nothing else."
)


class SimpleRagAgent:

    def __init__(
        self,
        ep:  ProcessorEndpoint,
        stt: SttClient,
        vlm: VlmClient,
        tts: TtsClient,
        rag: RagClient,
        *,
        top_k:                  int   = 4,
        default_prompt:         str   = "Describe what you see.",
        system_prompt:          str   = DEFAULT_SYSTEM_PROMPT,
        silence_threshold:      float = 0.01,
        silence_duration:       float = 0.8,
        min_speech:             float = 0.3,
        frame_max_age_s:        float = 2.0,
        camera_on_timeout_s:    float = 15.0,
        camera_grace_s:         float = 5.0,
        doc_skip_camera_score:  float = 0.35,
    ) -> None:
        self._ep  = ep
        self._stt = stt
        self._vlm = vlm
        self._tts = tts
        self._rag = rag
        self._top_k = top_k

        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_frame(self._on_frame)
        self._ep.on_participant(self._on_participant)

        self._default_prompt    = default_prompt
        self._system_prompt     = system_prompt
        self._vad_threshold     = silence_threshold
        self._vad_silence_s     = silence_duration
        self._vad_min_s         = min_speech
        self._frame_max_age_us     = int(frame_max_age_s * 1_000_000)
        self._camera_on_timeout    = camera_on_timeout_s
        self._camera_grace_s       = camera_grace_s
        self._doc_skip_camera_score = doc_skip_camera_score

        self._voice:  dict[str, VoiceState]              = {}
        self._latest: dict[tuple[str, str], FrameSignal] = {}

        # Camera on demand state
        self._camera_on: dict[str, bool]                 = {}
        self._camera_held: set[str]                      = set()
        self._camera_off_timers: dict[str, asyncio.Task] = {}
        self._frame_events: dict[str, asyncio.Event]     = {}

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

        # Speculative camera-on: start streaming before STT finishes.
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

        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()

        self._camera_held.add(pid)
        t0     = time.monotonic()
        status = "done"
        try:
            # Retrieve first — it's fast (~30 ms) and tells us whether the
            # question is doc-grounded, which determines whether we need to
            # bother with the camera at all.
            chunks = await self._safe_retrieve(query)
            top_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
            visual    = _is_visual_query(query) or text.lower().strip() == "ping"
            # Skip the camera path entirely when retrieval is confident AND
            # the query has no visual deictic words.  We still use a frame
            # if one is already cached and fresh, but we don't trigger
            # startCamera and we don't wait.
            skip_camera = (not visual) and (top_score >= self._doc_skip_camera_score)

            if skip_camera:
                cached = self._latest_signal(pid)
                sig = cached if (cached and self._is_fresh(cached)) else None
            else:
                sig = self._latest_signal(pid)
                if not (sig and self._is_fresh(sig)):
                    await self._ensure_camera_on(pid)
                    sig = await self._wait_for_camera_frame(pid, self._camera_on_timeout)
                    if sig is None:
                        # Reset so next query re-sends startCamera.
                        self._camera_on[pid] = False

            frame     = await self._ep.request_frame(sig) if sig else None
            image_url = encode_image(frame_to_pil(frame)) if frame else None
            reply_pts = frame.pts_us if frame else pts_us

            user_text = _build_user_message(chunks, query)
            if image_url is not None:
                logger.info(
                    "vlm+rag  pid={!r}  {}x{}  chunks={}  query={!r}",
                    pid, frame.width, frame.height, len(chunks), query[:60],
                )
            else:
                logger.info(
                    "vlm+rag text-only  pid={!r}  chunks={}  query={!r}",
                    pid, len(chunks), query[:60],
                )

            await self._ep.set_status("processing", pid)
            try:
                full_response = await self._stream_and_speak(
                    pid, image_url, user_text, reply_pts,
                )
            finally:
                await self._ep.set_status("idle", pid)

            if full_response is not None:
                await self._reply(pid, full_response, reply_pts)

        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            self._camera_held.discard(pid)
            self._schedule_camera_off(pid)
            print_task_done_banner(
                "simple-rag-example",
                status=status,
                detail=f"pid={pid!r}  query={query[:60]!r}",
                duration_s=time.monotonic() - t0,
            )

    async def _safe_retrieve(self, query: str) -> list[dict]:
        """Retrieve doc chunks; return empty list on any failure."""
        try:
            return await self._rag.retrieve(query, top_k=self._top_k)
        except Exception as exc:
            logger.warning("rag-mcp retrieve failed: {} — proceeding without context", exc)
            return []

    async def _stream_and_speak(
        self, pid: str, image_url: str | None, user_text: str, fallback_pts_us: int,
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
                    image_url, user_text, system_prompt=self._system_prompt,
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
            # (speculative _on_audio + _handle_query) can't both send startCamera.
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
        """Wait up to ``timeout`` seconds for any new FrameSignal for ``pid``."""
        ev = self._frame_events.setdefault(pid, asyncio.Event())
        t0 = asyncio.get_event_loop().time()
        deadline = t0 + timeout

        # TOCTOU: clear event, then re-check before blocking.
        # IMPORTANT: only accept a signal that is *fresh*.  A stale FrameSignal
        # from a track that has since stopped will still live in self._latest;
        # using it makes request_frame return an 8x8 placeholder because the
        # underlying track is gone.
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
        """Schedule stopCamera for ``pid`` after the grace period."""
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
            logger.opt(exception=True).error("tts error pid={!r}: {}", pid, exc)

    # ── frame tracking ────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        prev = self._latest.get((sig.participant_id, sig.track_id))
        self._latest[(sig.participant_id, sig.track_id)] = sig
        if prev is None:
            logger.info(
                "first frame signal  pid={!r}  track={}  age_ms={:.0f}",
                sig.participant_id, sig.track_id,
                (now_us() - sig.pts_us) / 1_000,
            )
        ev = self._frame_events.get(sig.participant_id)
        if ev is not None:
            ev.set()

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        # Use pts_us (real Unix timestamp) not seq — seq restarts on camera restart.
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

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            await self._ep.run()
        finally:
            for t in self._camera_off_timers.values():
                t.cancel()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


# Phrases that signal the user is asking about the live scene (i.e. the
# camera frame is required).  When a query contains any of these the
# agent will use the camera path; otherwise, if retrieval returns a
# confidently-scored chunk, the camera path is skipped entirely.
# Conservative list — false positive (unnecessary wait) is preferable
# to false negative (missing the camera when it was needed).
_VISUAL_HINT_PATTERNS = (
    " this", "this ", " that ", "what is this", "what's this",
    "what am i", "what do you see", "what does it look",
    "describe what", "what color", "what's the color",
    "look at", "looking at", "holding", "in front of me",
    "what's in", "what is in",
)


def _is_visual_query(text: str) -> bool:
    """True if the query likely needs the camera frame to be answered."""
    t = " " + text.lower().strip() + " "
    return any(p in t for p in _VISUAL_HINT_PATTERNS)


def _build_user_message(chunks: list[dict], query: str) -> str:
    """Assemble the user-turn text: retrieved context (if any) + the query.

    When chunks are present each one is prefixed with its source filename so
    the model can attribute answers.  When no chunks were retrieved, the query
    is sent as-is — no fabricated context, no apology for missing docs.
    """
    if not chunks:
        return query

    parts = [
        f"[{i}] source={c['source']}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    context = "\n\n".join(parts)
    return (
        f"Reference text:\n\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer the user question above using the reference text.  Do not "
        f"describe the camera scene unless the question is explicitly about "
        f"what is visible (e.g. \"what am I holding\", \"what do you see\")."
    )
