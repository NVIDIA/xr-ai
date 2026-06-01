# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice conversation loop shared by agent workers.

Owns: per-pid VAD lifecycle, STT, voice-gate wiring, per-pid in-flight
cancel/flush, sentence-batched streaming TTS, return-audio fan-out, and
the participant-join/leave hooks. The worker supplies a
``ProcessorEndpoint``, STT/TTS services, a ``VoiceGateConfig``, and one
``on_query`` callback that returns either a ``str`` (one-shot reply) or
an ``AsyncIterator[str]`` (streamed tokens, sentence-batched downstream).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from xr_ai_agent import (
    AudioChunk, DataMessage, ParticipantEvent, ProcessorEndpoint,
)
from xr_ai_models import STTService, TTSService
from xr_ai_vad import VadDetector
from xr_ai_voicegate import AudioSink, VoiceGate, VoiceGateConfig

from .audio import int16_pcm_to_wav, now_us, wav_to_chunks
from .gate_wiring import GateBindings, QueryHandler, QueryResult
from .state import VoiceState
from .streaming import stream_text_to_audio


logger = logging.getLogger("xr_ai_conversation")


@dataclass(frozen=True)
class VadConfig:
    silence_duration: float = 0.8
    min_speech:       float = 0.3
    silero_threshold: float = 0.5


class _EpAudioSink:
    """Adapter exposing ``ProcessorEndpoint`` as an :class:`AudioSink`.

    Explodes a WAV blob into 20 ms ``AudioChunk``s and streams them
    through ``send_return_audio`` so the voice gate, the streaming TTS
    helper, and the loop's own ``say()`` all push audio through one
    code path."""

    def __init__(self, ep: ProcessorEndpoint) -> None:
        self._ep = ep

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        for chunk in wav_to_chunks(wav_bytes, pid):
            await self._ep.send_return_audio(chunk)


class ConversationLoop:
    """Voice-conversation runtime.

    Wires the audio → VAD → STT → voice-gate → ``on_query`` → TTS →
    return-audio path for any worker that consumes the hub's mic feed.
    Construct once per worker, register optional hooks, then ``await
    run()``. Per-pid in-flight cancel/flush keeps a new query from
    racing with a previous in-flight response.

    The ``on_query`` callback may return either a ``str`` (canned reply
    or one-shot LLM/VLM call) or an ``AsyncIterator[str]`` (streamed
    tokens — sentences are peeled and synthesized in parallel). Both
    forms also send the final assembled text on ``text_topic`` for
    clients that listen on the data channel.
    """

    def __init__(
        self,
        *,
        ep:             ProcessorEndpoint,
        stt:            STTService,
        tts:            TTSService,
        voice_gate_cfg: VoiceGateConfig,
        vad_cfg:        VadConfig,
        on_query:       QueryHandler,
        on_speech_start:       Callable[[str], Awaitable[None]] | None = None,
        on_participant_joined: Callable[[str], Awaitable[None]] | None = None,
        on_participant_left:   Callable[[str], Awaitable[None]] | None = None,
        on_stop_extra:         Callable[[str], Awaitable[None]] | None = None,
        text_topic: str = "agent.response",
        greeting:   str | Callable[[VoiceGate], str] | None = None,
    ) -> None:
        self._ep         = ep
        self._stt        = stt
        self._tts        = tts
        self._vad_cfg    = vad_cfg
        self._text_topic = text_topic

        self._bindings = GateBindings(
            on_query              = on_query,
            on_stop_extra         = on_stop_extra,
            on_participant_joined = on_participant_joined,
            on_participant_left   = on_participant_left,
        )
        self._on_speech_start = on_speech_start
        self._greeting        = greeting

        self._voice: dict[str, VoiceState] = {}

        self._audio_sink = _EpAudioSink(ep)
        self._gate = VoiceGate(
            voice_gate_cfg,
            audio_sink = self._audio_sink,
            tts        = tts,
        )
        self._gate.on_query(self._on_gate_query)
        self._gate.on_stop(self._handle_stop)
        self._gate.on_phrase_only(self._on_phrase_only)
        self._gate.on_drop(self._on_drop)
        self._gate.on_participant_joined(self._greet)

        ep.on_audio(self._on_audio)
        ep.on_participant(self._on_participant)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()

    # ── audio path: PCM → VAD → STT → gate ────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        vs = self._get_voice(chunk.participant_id)
        assert vs.vad is not None
        # Hub delivers float32 LE PCM; VadDetector takes int16 LE PCM.
        f32 = np.frombuffer(chunk.data, dtype=np.float32)
        i16 = (np.clip(f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        await vs.vad.feed(i16, chunk.sample_rate)

    def _get_voice(self, pid: str) -> VoiceState:
        vs = self._voice.get(pid)
        if vs is None:
            vs = VoiceState()
            vs.vad = VadDetector(
                on_utterance     = lambda audio, sr, _pid=pid: self._on_vad_utterance(_pid, audio, sr),
                on_speech_start  = lambda _pid=pid: self._on_vad_speech_start(_pid),
                silence_duration = self._vad_cfg.silence_duration,
                min_speech       = self._vad_cfg.min_speech,
                silero_threshold = self._vad_cfg.silero_threshold,
            )
            self._voice[pid] = vs
        return vs

    async def _on_vad_speech_start(self, pid: str) -> None:
        if self._on_speech_start is None:
            return
        try:
            await self._on_speech_start(pid)
        except Exception:
            logger.exception("on_speech_start hook raised pid=%r", pid)

    async def _on_vad_utterance(self, pid: str, audio_bytes: bytes, sample_rate: int) -> None:
        vs = self._voice.get(pid)
        if vs is None or vs.transcribing:
            return
        vs.transcribing = True
        try:
            wav  = int16_pcm_to_wav(audio_bytes, sample_rate)
            text = (await self._stt.transcribe(wav)).strip()
            if not text:
                return
            await self._gate.feed(pid, text)
        except Exception:
            logger.exception("stt error pid=%r", pid)
        finally:
            vs.transcribing = False

    # ── voice-gate handlers ───────────────────────────────────────────────────

    async def _on_gate_query(self, pid: str, query: str, fresh_match: bool) -> None:
        if fresh_match:
            asyncio.create_task(self._gate.play_chime(pid))
        await self._dispatch_internal(pid, query, pts_us=now_us(), fresh_match=fresh_match)

    async def _on_phrase_only(self, pid: str) -> None:
        # The chime here acknowledges that the wake word was heard while
        # the follow-up window is open; without it the user has no signal
        # that the gate is ready for the next utterance.
        await self._gate.play_chime(pid)

    async def _on_drop(self, pid: str, text: str) -> None:
        # The gate already logged; nothing for the loop to do.
        return

    # ── interruptible dispatch ────────────────────────────────────────────────

    async def dispatch(self, pid: str, text: str, *, pts_us: int) -> None:
        """Cancel any in-flight response for ``pid``, flush queued audio,
        then start the new query as a tracked task.

        Data-channel callers use this directly; the gate path goes
        through ``_dispatch_internal`` so it can pass the gate's
        ``fresh_match`` flag through to the user's ``on_query``."""
        await self._dispatch_internal(pid, text, pts_us=pts_us, fresh_match=True)

    async def _dispatch_internal(
        self, pid: str, text: str, *, pts_us: int, fresh_match: bool,
    ) -> None:
        vs = self._get_voice(pid)
        async with vs.dispatch_lock:
            await self._cancel_inflight_locked(vs, pid)
            await self._ep.flush_return_audio(pid)
            vs.current_task = asyncio.create_task(
                self._run_query(pid, text, pts_us, fresh_match),
                name=f"conv-query-{pid}",
            )

    async def _cancel_inflight_locked(self, vs: VoiceState, pid: str) -> None:
        old = vs.current_task
        if old is None or old.done():
            return
        logger.info("interrupt pid=%r — cancelling in-flight response", pid)
        old.cancel()
        try:
            await old
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("in-flight task error during cancel pid=%r", pid)

    async def _run_query(
        self, pid: str, text: str, pts_us: int, fresh_match: bool,
    ) -> None:
        try:
            result = await self._bindings.on_query(pid, text, fresh_match)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("on_query raised pid=%r", pid)
            return

        try:
            if isinstance(result, str):
                await self._speak(pid, result)
                final_text = result
            else:
                final_text = await stream_text_to_audio(
                    tokens     = result,
                    pid        = pid,
                    tts        = self._tts,
                    audio_sink = self._audio_sink,
                    on_wav     = self._gate.observe_tts_wav,
                )
        except asyncio.CancelledError:
            raise

        if final_text:
            await self._reply_data(pid, final_text, pts_us)

    # ── outbound helpers ──────────────────────────────────────────────────────

    async def say(self, pid: str, text: str) -> None:
        """Synthesize ``text`` and play it on the return-audio channel.

        Does not touch the data channel. Use :meth:`reply` for a paired
        data + audio response.
        """
        await self._speak(pid, text)

    async def reply(self, pid: str, text: str, *, pts_us: int) -> None:
        """Send ``text`` on the data channel AND speak it on return audio."""
        await self._reply_data(pid, text, pts_us)
        await self._speak(pid, text)

    async def _speak(self, pid: str, text: str) -> None:
        if not text:
            return
        try:
            wav = await self._tts.synthesize(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("tts synthesize error pid=%r", pid)
            return
        self._gate.observe_tts_wav(wav)
        try:
            await self._audio_sink.play_wav(pid, wav)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("return-audio play_wav error pid=%r", pid)

    async def _reply_data(self, pid: str, text: str, pts_us: int) -> None:
        await self._ep.send_return_data(DataMessage(
            participant_id = pid,
            topic          = self._text_topic,
            pts_us         = pts_us,
            data           = text.encode(),
        ))

    # ── stop ──────────────────────────────────────────────────────────────────

    async def _handle_stop(self, pid: str) -> None:
        """Cancel + flush + ``say_stop_ack`` + ``on_stop_extra`` + data echo.

        Order matches the design spec (PR-A) — it differs from the
        original simple-vlm-example, which echoed the data message
        before the stop-ack TTS; this loop runs the stop-ack first so
        the canned audio starts playing while the data echo travels."""
        vs = self._voice.get(pid)
        if vs is not None:
            async with vs.dispatch_lock:
                await self._cancel_inflight_locked(vs, pid)
                await self._ep.flush_return_audio(pid)
                vs.current_task = None

        await self._gate.say_stop_ack(pid)

        if self._bindings.on_stop_extra is not None:
            try:
                await self._bindings.on_stop_extra(pid)
            except Exception:
                logger.exception("on_stop_extra hook raised pid=%r", pid)

        await self._reply_data(pid, "Okay, I will stop.", now_us())

    # ── participant lifecycle ─────────────────────────────────────────────────

    async def _on_participant(self, event: ParticipantEvent) -> None:
        pid = event.participant_id
        if event.joined:
            if self._bindings.on_participant_joined is not None:
                try:
                    await self._bindings.on_participant_joined(pid)
                except Exception:
                    logger.exception("on_participant_joined hook raised pid=%r", pid)
            asyncio.create_task(self._gate.participant_joined(pid))
            return

        vs = self._voice.pop(pid, None)
        if vs is not None and vs.current_task is not None and not vs.current_task.done():
            vs.current_task.cancel()
        self._gate.forget(pid)
        if self._bindings.on_participant_left is not None:
            try:
                await self._bindings.on_participant_left(pid)
            except Exception:
                logger.exception("on_participant_left hook raised pid=%r", pid)

    async def _greet(self, pid: str) -> None:
        text = self._resolve_greeting()
        if not text:
            return
        try:
            await self._speak(pid, text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("greet error pid=%r", pid)

    def _resolve_greeting(self) -> str:
        g = self._greeting
        if g is None:
            help_text = self._gate.format_phrase_help()
            if help_text is None:
                return "Hi, I'm listening. Ask me anything."
            return f"Hi, I'm listening. {help_text}"
        if isinstance(g, str):
            return g
        try:
            return g(self._gate) or ""
        except Exception:
            logger.exception("greeting callable raised")
            return ""
