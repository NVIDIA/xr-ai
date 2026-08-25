# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VoiceGateProcessor`` — wraps :class:`xr_ai_voicegate.VoiceGate` as a
pipecat ``FrameProcessor``.

Maps gate events to frames:

- ``on_query(pid, text, fresh_match)``    → ``GatedQueryFrame``
- ``on_stop(pid)``                        → ``InterruptionFrame`` + bounded stop-ack text
- ``on_phrase_only(pid)``                 → no frame (internal state only)
- ``on_participant_joined(pid)``          → bounded greeting text (only when ``format_phrase_help`` returns text)

The processor also acts as the gate's ``AudioSink`` so the chime and
stop-ack play out via the same audio path as TTS. Partial transcripts can
acknowledge a wake phrase before the complete command is available.
"""
from __future__ import annotations

import time

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TextFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_models import TTSService
from xr_ai_voicegate import VoiceGate, VoiceGateConfig

from .._audio import wav_to_output_frames
from .._frames import (
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
    TextResponseEndFrame,
)


_STOP_ACK_TEXT = "Okay, I will stop."


class VoiceGateProcessor(FrameProcessor):
    """Adapts ``VoiceGate`` to pipecat frames.

    Owns the gate's handler bindings; emits the right frames downstream
    when gate events fire. Doubles as the gate's ``AudioSink`` so the
    chime and ``say_stop_ack`` WAVs travel the same pipeline route as
    real TTS audio.
    """

    def __init__(
        self,
        *,
        cfg: VoiceGateConfig,
        tts: TTSService,
        gate: VoiceGate | None = None,
    ) -> None:
        """Build the gate-backed processor.

        ``cfg`` and ``tts`` are the usual entry path — the gate is built
        with this processor as its ``AudioSink`` so the chime / stop-ack
        WAVs route back through the same pipeline.

        ``gate`` is an escape hatch for tests that want to pre-build the
        gate with a custom sink or TTS double. When supplied, ``cfg`` /
        ``tts`` are not used to construct a new gate — the caller owns
        the bindings.
        """
        super().__init__()
        self._gate = gate or VoiceGate(cfg, audio_sink=self, tts=tts)
        self._gate.bind(
            on_query              = self._on_gate_query,
            on_stop               = self._on_gate_stop,
            on_phrase_only        = self._on_gate_phrase_only,
            on_participant_joined = self._on_gate_participant_joined,
        )
        self._early_wake_ack: set[str] = set()
        self._feeding_speech_transcript = False
        # Speech-onset timestamp (µs) of the transcript currently being fed to
        # the gate. ``VoiceGate.feed`` invokes ``_on_gate_query`` synchronously,
        # so the value is read back inside that callback.
        self._feeding_pts_us: int | None = None

    @property
    def gate(self) -> VoiceGate:
        """The wrapped gate — exposed so the factory can hand it to
        :class:`StreamingTtsProcessor` for ``observe_tts_wav`` callbacks."""
        return self._gate

    @property
    def early_wake_ack_enabled(self) -> bool:
        """Whether partial STT should probe for an early wake acknowledgement."""
        return self._gate.wake_ack_enabled

    async def handle_partial_transcript(self, pid: str, text: str) -> bool:
        """Handle a partial transcript and acknowledge a complete wake phrase.

        True means acknowledged and False means the probe needs more audio.
        A later sentence can introduce a wake phrase even when the current
        partial transcript is not a phrase prefix.
        """
        if self._gate.matches_magic_phrase(text):
            await self._emit_chime(pid, early=True)
            return True
        return False

    # ── AudioSink ─────────────────────────────────────────────────────────────

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        """``AudioSink`` impl — VoiceGate calls this for chime + stop-ack.

        We decode the WAV into 20 ms chunks and push them as
        ``OutputAudioRawFrame``s, matching the same int16 PCM path TTS
        uses. ``transport_destination`` carries the pid so the output
        transport knows which participant to send the audio back to.
        """
        try:
            frames = wav_to_output_frames(wav_bytes, pid)
        except Exception:
            logger.exception("voice-gate audio sink decode failed pid={!r}", pid)
            return
        for out in frames:
            await self.push_frame(out)
        if frames:
            stopped = TTSStoppedFrame()
            stopped.transport_destination = pid
            await self.push_frame(stopped)

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # ``pts`` carries the hub capture time of the audio that opened this
            # utterance (nanoseconds); the gate's query callback stamps it onto
            # the turn so downstream consumers anchor to when the user spoke.
            self._feeding_pts_us = frame.pts // 1_000 if frame.pts is not None else None
            self._feeding_speech_transcript = bool(
                frame.transport_source and frame.transport_source == frame.user_id
            )
            try:
                await self._gate.feed(frame.user_id, frame.text)
            finally:
                self._feeding_pts_us = None
                self._feeding_speech_transcript = False
                self._early_wake_ack.discard(frame.user_id)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            if frame.transport_source:
                self._early_wake_ack.discard(frame.transport_source)
                self._gate.begin_utterance(frame.transport_source)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantJoinedFrame):
            await self._gate.participant_joined(frame.participant_id)
            # Pass the frame through so assistants/other processors can hook in.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            self._gate.forget(frame.participant_id)
            self._early_wake_ack.discard(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── gate handlers ─────────────────────────────────────────────────────────

    async def _on_gate_query(self, pid: str, text: str, fresh_match: bool) -> None:
        if fresh_match and not self._feeding_speech_transcript:
            await self._emit_chime(pid, early=False)
        await self.push_frame(GatedQueryFrame(
            participant_id = pid,
            text           = text,
            fresh_match    = fresh_match,
            # Speech onset when the transport supplied it; wall clock only as a
            # fallback for inputs that carry no capture time (e.g. a synthetic
            # transcript injected without an originating audio frame).
            pts_us         = (
                self._feeding_pts_us
                if self._feeding_pts_us is not None
                else time.time_ns() // 1_000
            ),
        ))

    async def _on_gate_stop(self, pid: str) -> None:
        logger.info("stop ack emit pid={!r}", pid)
        f = InterruptionFrame()
        f.transport_source = pid
        await self.push_frame(f)
        await self._emit_text_response(pid, _STOP_ACK_TEXT)

    async def _on_gate_phrase_only(self, pid: str) -> None:
        await self._emit_chime(pid, early=False)

    async def _emit_chime(self, pid: str, *, early: bool) -> None:
        """Play at most one chime for a wake-matched utterance."""
        if pid in self._early_wake_ack:
            return
        logger.info("chime fire pid={!r} early={}", pid, early)
        try:
            emitted = await self._gate.play_chime(pid)
        except Exception:
            logger.exception("voicegate chime emit failed pid={!r}", pid)
            return
        if emitted and early:
            self._early_wake_ack.add(pid)

    async def _on_gate_participant_joined(self, pid: str) -> None:
        greeting = self._gate.format_phrase_help()
        if not greeting:
            # Always-on mode: surfacing a stock greeting "Hi, I'm
            # listening" would be intrusive on samples that never opted
            # into a wake word, so stay silent.
            return
        logger.info("greeting emit pid={!r}", pid)
        await self._emit_text_response(pid, greeting)

    async def _emit_text_response(self, pid: str, text: str) -> None:
        """Emit one addressed utterance and its synthesis-order boundary."""
        frame = TextFrame(text=text)
        frame.transport_destination = pid
        await self.push_frame(frame)
        await self.push_frame(TextResponseEndFrame(pid=pid))
