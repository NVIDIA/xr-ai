# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VoiceGateProcessor`` — wraps :class:`xr_ai_voicegate.VoiceGate` as a
pipecat ``FrameProcessor``.

Maps gate events to frames:

- ``on_query(pid, text, fresh_match)``    → ``GatedQueryFrame``
- ``on_stop(pid)``                        → ``InterruptionFrame`` + ``TextFrame("Okay, I will stop.")``
- ``on_phrase_only(pid)``                 → no frame (internal state only)
- ``on_participant_joined(pid)``          → ``TextFrame`` with the greeting (only when ``format_phrase_help`` returns text)

The processor also acts as the gate's ``AudioSink`` so the chime and
stop-ack play out via the same audio path as TTS. The chime fires on
fresh magic-phrase matches.
"""
from __future__ import annotations

import time

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_models import TTSService
from xr_ai_voicegate import VoiceGate, VoiceGateConfig

from ..audio import wav_to_output_frames
from ..frames import GatedQueryFrame, ParticipantJoinedFrame, ParticipantLeftFrame


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

    @property
    def gate(self) -> VoiceGate:
        """The wrapped gate — exposed so the factory can hand it to
        :class:`StreamingTtsProcessor` for ``observe_tts_wav`` callbacks."""
        return self._gate

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

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._gate.feed(frame.user_id, frame.text)
            return

        if isinstance(frame, ParticipantJoinedFrame):
            await self._gate.participant_joined(frame.participant_id)
            # Pass the frame through so brains/other processors can hook in.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            self._gate.forget(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── gate handlers ─────────────────────────────────────────────────────────

    async def _on_gate_query(self, pid: str, text: str, fresh_match: bool) -> None:
        if fresh_match:
            # Fire-and-forget: chime is a UX nicety; failure to play
            # must not delay the query dispatch downstream.
            logger.info("chime fire pid={!r}", pid)
            try:
                await self._gate.play_chime(pid)
            except Exception:
                logger.exception("voicegate chime emit failed pid={!r}", pid)
        await self.push_frame(GatedQueryFrame(
            participant_id = pid,
            text           = text,
            fresh_match    = fresh_match,
            pts_us         = time.time_ns() // 1_000,
        ))

    async def _on_gate_stop(self, pid: str) -> None:
        logger.info("stop ack emit pid={!r}", pid)
        f = InterruptionFrame()
        f.transport_source = pid
        await self.push_frame(f)
        ack = TextFrame(text=_STOP_ACK_TEXT)
        ack.transport_destination = pid
        await self.push_frame(ack)

    async def _on_gate_phrase_only(self, pid: str) -> None:
        # No frame is emitted — the gate's internal followup state is
        # the only effect. The hook exists so a future debug path can
        # observe phrase-only matches without changing the contract.
        return

    async def _on_gate_participant_joined(self, pid: str) -> None:
        greeting = self._gate.format_phrase_help()
        if not greeting:
            # Always-on mode: surfacing a stock greeting "Hi, I'm
            # listening" would be intrusive on samples that never opted
            # into a wake word, so stay silent.
            return
        logger.info("greeting emit pid={!r}", pid)
        frame = TextFrame(text=greeting)
        frame.transport_destination = pid
        await self.push_frame(frame)
