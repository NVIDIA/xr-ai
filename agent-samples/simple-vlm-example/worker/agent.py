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

* `vlm.response` data message — the assembled text reply.
* `xr-hub-return-{pid}` audio track — sentence-by-sentence Piper TTS,
  started in parallel as soon as each sentence completes.

Interruption
------------
A new query (audio utterance, data message, or ping) cancels any
in-flight response for the same participant.  The dispatcher cancels
the running task, awaits its cleanup, and then calls
``flush_return_audio`` so already-queued TTS chunks are dropped at the
hub before the new response starts.  Dispatch is serialised per pid via
a lock so rapid-fire queries don't race over ``current_task``.
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx

from xr_ai_agent import (AudioChunk, DataMessage, FrameSignal,
                          ParticipantEvent, ProcessorEndpoint)

from audio import chunks_to_wav, now_us, rms, wav_to_chunks
from pixels import encode_image, frame_to_pil
from services import SttClient, TtsClient, VlmClient
from voice import VoiceState

log = logging.getLogger("simple_vlm_example")


DEFAULT_SYSTEM_PROMPT = (
    "You are an XR assistant looking at the user's live camera feed. "
    "Help them understand what they see in their environment.\n"
    "\n"
    "Style:\n"
    "- Reply in plain conversational English — never JSON, code, or markdown.\n"
    "- Keep replies to 10-15 words by default. Only go longer when the user "
    "explicitly asks for detail (e.g. 'describe in detail', 'tell me more', "
    "'elaborate', 'explain').\n"
    "- If the user says 'stop', tells you to be quiet, or asks you to stop "
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
        default_prompt:    str   = "Describe what you see.",
        system_prompt:     str   = DEFAULT_SYSTEM_PROMPT,
        silence_threshold: float = 0.01,
        silence_duration:  float = 0.8,
        min_speech:        float = 0.3,
    ) -> None:
        self._ep  = ep
        self._stt = stt
        self._vlm = vlm
        self._tts = tts

        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_frame(self._on_frame)
        self._ep.on_participant(self._on_participant)

        self._default_prompt  = default_prompt
        self._system_prompt   = system_prompt
        self._vad_threshold   = silence_threshold
        self._vad_silence_s   = silence_duration
        self._vad_min_s       = min_speech

        self._voice:  dict[str, VoiceState]                = {}
        self._latest: dict[tuple[str, str], FrameSignal]   = {}

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
            # Sub-min-speech blip followed by silence — drop so it doesn't
            # pollute the next real utterance.
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
            log.info("audio query  pid=%r  %r", pid, text[:80])
            await self._dispatch_query(pid, text, pts_us=now_us())
        except httpx.HTTPError as exc:
            log.error("stt error pid=%r: %s", pid, exc)
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
        log.info("data query  pid=%r  %r", msg.participant_id, text[:80])
        await self._dispatch_query(msg.participant_id, text, pts_us=msg.pts_us)

    # ── interruptable query dispatch ──────────────────────────────────────────

    async def _dispatch_query(self, pid: str, text: str, *, pts_us: int) -> None:
        """Cancel any in-flight response for ``pid``, flush queued audio,
        then start the new query as a tracked task.

        The flush runs **unconditionally** — not only on mid-response
        interruption.  When the VLM is fast, the old task may have already
        finished (``current_task.done()``) while several seconds of TTS
        audio is still draining through the hub's pacing pipe → LiveKit
        → client jitter buffer.  Without flushing in that window, the
        new query's audio queues *behind* the old, and the user hears the
        old reply continue uninterrupted before the new one starts.
        """
        vs = self._get_voice(pid)

        async with vs.dispatch_lock:
            old = vs.current_task
            if old is not None and not old.done():
                log.info("interrupt pid=%r — cancelling in-flight response", pid)
                old.cancel()
                try:
                    await old
                except (asyncio.CancelledError, Exception):
                    pass

            # Always flush: covers both cancelled-mid-response and
            # finished-but-audio-still-playing cases.  Cheap O(1) when there
            # is nothing queued.
            await self._ep.flush_return_audio(pid)

            vs.current_task = asyncio.create_task(self._handle_query(pid, text, pts_us))

    async def _handle_query(self, pid: str, text: str, pts_us: int) -> None:
        # "ping" — use the configured default prompt against the latest frame.
        # ("stop" is handled by the system prompt — the unconditional flush
        # in _dispatch_query has already silenced any in-flight TTS by then,
        # so the model just replies with a short acknowledgement.)
        query = self._default_prompt if text.lower().strip() == "ping" else text

        sig = self._latest_signal(pid)
        if sig is None:
            log.warning("query from %r — no video frame yet", pid)
            await self._say(pid, "No video frame available yet.", pts_us)
            return

        frame = await self._ep.request_frame(sig)
        if frame is None:
            await self._say(pid, "Frame data unavailable — please retry.", pts_us)
            return

        image_url = encode_image(frame_to_pil(frame))
        log.info("vlm  pid=%r  %dx%d  query=%r", pid, frame.width, frame.height, query[:60])

        await self._ep.set_status("processing", pid)
        try:
            full_response = await self._stream_and_speak(pid, image_url, query, frame.pts_us)
        finally:
            # Always restore status — both on success and on interruption.
            await self._ep.set_status("idle", pid)

        if full_response is not None:
            await self._reply(pid, full_response, frame.pts_us)

    async def _stream_and_speak(
        self, pid: str, image_url: str, query: str, fallback_pts_us: int,
    ) -> str | None:
        """Run streaming VLM → sentence-batched TTS in parallel.

        Returns the assembled response text on success, or ``None`` on VLM
        error (the error reply has already been sent).  On task cancellation
        (interruption), cancels all pending TTS work and re-raises.
        """
        full_response = ""
        sentence_buf  = ""
        # Queue of synthesis tasks in sentence order.  None signals normal end.
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
                    log.error("tts audio error pid=%r: %s", pid, exc, exc_info=True)

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
                log.error("vlm-server error: %s", exc)
                await tts_queue.put(None)
                await sender
                await self._reply(pid, "VLM server unavailable — please retry.", fallback_pts_us)
                return None

            await tts_queue.put(None)
            await sender
            full_response = full_response.strip()
            log.info("vlm response  pid=%r  %d chars", pid, len(full_response))
            return full_response

        except asyncio.CancelledError:
            # Interrupted: cancel synth tasks + sender, swallow their exceptions,
            # and re-raise so the caller (dispatcher) sees the cancellation.
            log.info("response cancelled pid=%r", pid)
            for t in pending_synth:
                t.cancel()
            sender.cancel()
            for t in (*pending_synth, sender):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            raise

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
            log.error("tts error pid=%r: %s", pid, exc, exc_info=True)

    # ── frame tracking ────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._latest[(sig.participant_id, sig.track_id)] = sig

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.seq)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            return
        pid = event.participant_id
        vs  = self._voice.pop(pid, None)
        if vs is not None and vs.current_task is not None and not vs.current_task.done():
            vs.current_task.cancel()
        for k in [k for k in self._latest if k[0] == pid]:
            del self._latest[k]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()
