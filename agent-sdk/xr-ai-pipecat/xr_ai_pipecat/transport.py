# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XR-Media-Hub transport for Pipecat.

Bridges ``ProcessorEndpoint`` (ZMQ IPC) to Pipecat's frame pipeline.

Input  — float32 audio chunks from the hub at any sample rate, resampled
         to 16 kHz int16 ``InputAudioRawFrame`` for the STT processor.
Output — int16 PCM frames written by the TTS processor are converted back
         to float32 ``AudioChunk``s and pushed via ``send_return_audio``.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from loguru import logger
from scipy.signal import resample_poly
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from xr_ai_agent import (
    AudioChunk,
    DataMessage,
    ParticipantEvent,
    ProcessorEndpoint,
    Subscribe,
)

from .frames import ParticipantJoinedFrame, ParticipantLeftFrame

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

SAMPLE_RATE            = 16_000
NUM_CHANNELS           = 1
TTS_NATIVE_SAMPLE_RATE = 22_050


def _float32_to_int16(data: bytes) -> bytes:
    f32 = np.frombuffer(data, dtype=np.float32)
    return np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def _int16_to_float32(data: bytes) -> bytes:
    i16 = np.frombuffer(data, dtype=np.int16)
    return (i16.astype(np.float32) / 32767.0).tobytes()


def _hub_pcm_to_mono_16k(pcm_int16: bytes, channels: int, sample_rate: int) -> bytes:
    """Convert hub int16 PCM to mono 16 kHz int16 for the (mono) STT path.

    The hub delivers *interleaved* samples for multi-channel audio (L R L R …).
    Downmix to mono BEFORE resampling: passing an interleaved buffer straight to
    ``resample_poly`` treats it as one stream, mixing adjacent L/R samples and
    destroying channel alignment (#193). STT is mono, so we average channels.
    """
    if channels == 1 and sample_rate == SAMPLE_RATE:
        return pcm_int16  # common case — already mono 16 kHz, no work
    audio = np.frombuffer(pcm_int16, dtype=np.int16)
    if channels > 1:
        # Each interleaved frame is `channels` int16 samples; a complete hub
        # chunk is always a whole number of frames. Truncate any trailing
        # partial frame defensively so reshape can't raise on a malformed chunk.
        usable = (audio.size // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio.astype(np.float64), SAMPLE_RATE, sample_rate)
    return np.clip(np.round(audio), -32768, 32767).astype(np.int16).tobytes()


# ── Input ─────────────────────────────────────────────────────────────────────

class XRMediaHubInputTransport(BaseInputTransport):
    """Hub → Pipecat: float32 hub audio → 16 kHz int16 pipecat frames."""

    def __init__(self, ep: ProcessorEndpoint, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._ep = ep
        self._ep_task: asyncio.Task | None = None
        self._started = False
        self._ep.on_audio(self._on_hub_audio)
        self._ep.on_participant(self._on_hub_participant)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._started = True
        self._ep_task = asyncio.create_task(self._ep.run(), name="ep-run")
        logger.info("XRMediaHubInputTransport started")

    async def stop(self, frame: EndFrame):
        self._started = False
        self._ep.stop()
        if self._ep_task:
            self._ep_task.cancel()
            try:
                await self._ep_task
            except asyncio.CancelledError:
                pass  # Expected: the task was explicitly cancelled above.
            self._ep_task = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._started = False
        self._ep.stop()
        if self._ep_task:
            self._ep_task.cancel()
        await super().cancel(frame)

    async def _on_hub_audio(self, chunk: AudioChunk) -> None:
        if not self._started:
            return
        pcm_int16 = _hub_pcm_to_mono_16k(
            _float32_to_int16(chunk.data), chunk.channels, chunk.sample_rate,
        )
        frame = InputAudioRawFrame(
            audio=pcm_int16,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        # pipecat's ``transport_source`` is the standard "which input
        # track did this come from" hook — set it to the hub-side
        # participant id so downstream processors (VadStt, brain, the
        # output transport's return_data / return_audio routing) can
        # address the right participant. Without this, every downstream
        # send falls back to the empty string and the hub drops the
        # message.
        frame.transport_source = chunk.participant_id
        await self.push_frame(frame)

    async def _on_hub_participant(self, event: ParticipantEvent) -> None:
        """Translate hub ``ParticipantEvent`` into pipecat lifecycle frames.

        The hub publishes one event per LiveKit join/leave; downstream
        processors (``VoiceGateProcessor`` greeting hook,
        ``BrainProcessor.set_target_participant``) consume the resulting
        ``ParticipantJoinedFrame`` / ``ParticipantLeftFrame``. Without
        this bridge the gate never greets and the brain never steers the
        output transport at a participant, so every TTS chunk is dropped
        by ``XRMediaHubOutputTransport.write_audio_frame``.

        Same ``_started`` guard as ``_on_hub_audio``: a late event after
        teardown is a no-op rather than racing the pipeline shutdown.
        """
        if not self._started:
            return
        if event.joined:
            await self.push_frame(
                ParticipantJoinedFrame(participant_id=event.participant_id),
            )
        else:
            await self.push_frame(
                ParticipantLeftFrame(participant_id=event.participant_id),
            )


# ── Output ────────────────────────────────────────────────────────────────────

class XRMediaHubOutputTransport(BaseOutputTransport):
    """Pipecat → Hub: int16 TTS frames → float32 ``AudioChunk``s."""

    def __init__(self, ep: ProcessorEndpoint, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._ep = ep
        self._target_participant: str = ""
        # Throttle the "no target participant" warning so a burst of
        # dropped audio frames produces one log line per burst rather
        # than one per frame. Reset when a target is set.
        self._missing_target_warned: bool = False
        # StartFrame stashed at start() so per-participant MediaSenders can
        # be created on demand (they need it to .start()).
        self._start_frame: StartFrame | None = None

    def set_target_participant(self, pid: str) -> None:
        # Retained as a fallback for frames that reach the output with no
        # ``transport_destination`` (routed through the default ``None``
        # sender). Per-participant routing now keys on the frame's own pid
        # (see ``write_audio_frame``), so this no longer has to be a single
        # room-wide target.
        logger.info("fallback target participant set pid={!r}", pid)
        self._target_participant = pid
        self._missing_target_warned = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Pipecat's BaseOutputTransport leaves the actual "register the
        # default media sender for destination=None" step to each
        # transport implementation — every shipped transport calls
        # set_transport_ready in its start() (see e.g. local/audio.py
        # and smallwebrtc/transport.py). Skipping it leaves
        # ``_media_senders`` empty so even a destination=None frame is
        # dropped at the router; combined with the upstream pid tagging
        # this was the silent audio-output drop.
        self._start_frame = frame
        await self.set_transport_ready(frame)

    async def _ensure_destination(self, pid: str) -> bool:
        """Lazily create a per-participant ``MediaSender`` keyed on ``pid``.

        Each participant gets its own sender so two participants' TTS streams
        never share one buffer (which would interleave their audio), and so
        ``write_audio_frame`` can read the sender-stamped
        ``transport_destination`` to address the return audio at the right
        participant. Returns ``True`` once a sender for ``pid`` exists.
        """
        if not pid:
            return False
        if pid in self._media_senders:
            return True
        if self._start_frame is None:
            return False
        sender = BaseOutputTransport.MediaSender(
            self,
            destination=pid,
            sample_rate=self.sample_rate,
            audio_chunk_size=self.audio_chunk_size,
            params=self._params,
        )
        await sender.start(self._start_frame)
        self._media_senders[pid] = sender
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        # Register a per-participant sender on join so audio addressed to that
        # pid has somewhere to route. Senders are not torn down on leave — one
        # idle sender per pid for the session is cheap, and tearing one down
        # mid-pipeline needs an EndFrame we don't have here.
        if isinstance(frame, ParticipantJoinedFrame):
            await self._ensure_destination(frame.participant_id)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def _handle_frame(self, frame: Frame) -> None:
        """Funnel every output frame through the default media sender.

        Pipecat's ``BaseOutputTransport._handle_frame`` routes a frame to
        ``_media_senders[frame.transport_destination]`` and drops it
        (with a warning) when the destination is not registered. Only the
        default ``None`` sender is registered by ``set_transport_ready``;
        upstream processors (``VoiceGateProcessor``,
        ``StreamingTtsProcessor``) tag outbound audio with
        ``transport_destination = pid`` so the hub knows which
        participant to send it back to. The two facts together used to
        drop every TTS / chime frame on the floor.

        Per-participant routing: each pid has its own ``MediaSender``
        (created on join, or lazily here). The frame keeps its
        ``transport_destination = pid`` so the router delivers it to that
        participant's sender, which stamps the pid back onto the chunk for
        ``write_audio_frame``. Only frames with no pid (or arriving before
        the sender could be created) fall back to the default ``None``
        sender + ``_target_participant``.
        """
        pid = frame.transport_destination
        if pid and pid not in self._media_senders:
            await self._ensure_destination(pid)
        if pid and pid not in self._media_senders:
            # Could not create a per-pid sender (no StartFrame yet) — fall back
            # to the default sender so the frame is not dropped at the router.
            # Null ``transport_destination`` only across the super() call, then
            # restore it so downstream taps/sinks still see which participant
            # the frame was addressed to (the save/restore intent main carried
            # before per-pid routing existed).
            frame.transport_destination = None
            await super()._handle_frame(frame)
            frame.transport_destination = pid
            return
        await super()._handle_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Pipecat's audio-out hook — invoked once per chunked output
        frame after the media sender has resampled and buffered.

        We forward the audio to the hub via ``send_return_audio``,
        addressing the configured target participant. Returns ``True`` so
        pipecat keeps pushing the frame downstream (any future tap /
        sink can still observe the audio); returns ``False`` only when
        no target participant is set — the hub would drop the message
        anyway, so we avoid emitting an unaddressable chunk.

        Note: the pipecat upstream method is ``write_audio_frame``
        (per-frame, returns bool), NOT ``write_raw_audio_frames`` —
        the previous implementation overrode a phantom name and pipecat
        never invoked it, which is why every TTS chunk was silently
        dropped before reaching the hub.
        """
        # Address the chunk at the participant whose sender produced it. The
        # per-pid MediaSender stamps ``transport_destination`` onto the frame;
        # the default sender leaves it None, so fall back to the room-wide
        # target for any unaddressed audio.
        pid = frame.transport_destination or self._target_participant
        if not pid:
            if not self._missing_target_warned:
                logger.warning(
                    "no target participant — dropping audio frame",
                )
                self._missing_target_warned = True
            return False
        pcm_float32 = _int16_to_float32(frame.audio)
        num_samples = len(frame.audio) // (2 * frame.num_channels)
        chunk = AudioChunk(
            pts_us=int(time.time() * 1_000_000),
            sample_rate=frame.sample_rate,
            channels=frame.num_channels,
            samples=num_samples,
            data=pcm_float32,
            participant_id=pid,
            track_id="tts",
        )
        await self._ep.send_return_audio(chunk)
        return True


# ── Transport wrapper ─────────────────────────────────────────────────────────

class XRMediaHubTransport(BaseTransport):
    """Owns the ProcessorEndpoint + bidirectional Pipecat transports."""

    def __init__(
        self,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)

        self._ep = ProcessorEndpoint(
            sub_addr=_HUB_PUB,
            push_addr=_HUB_PUSH,
            filter=Subscribe.AUDIO | Subscribe.DATA | Subscribe.VIDEO,
        )

        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_in_channels=NUM_CHANNELS,
            audio_out_enabled=True,
            audio_out_sample_rate=TTS_NATIVE_SAMPLE_RATE,
            audio_out_channels=NUM_CHANNELS,
        )

        self._input  = XRMediaHubInputTransport(self._ep, params, name=self._input_name)
        self._output = XRMediaHubOutputTransport(self._ep, params, name=self._output_name)
        self._target_participant: str = ""

    def input(self) -> XRMediaHubInputTransport:
        return self._input

    def output(self) -> XRMediaHubOutputTransport:
        return self._output

    @property
    def endpoint(self) -> ProcessorEndpoint:
        return self._ep

    async def send_return_data(self, msg: DataMessage) -> None:
        await self._ep.send_return_data(msg)

    @property
    def target_participant(self) -> str:
        return self._target_participant

    def set_target_participant(self, pid: str) -> None:
        self._target_participant = pid
        self._output.set_target_participant(pid)

    def cleanup_participant(self, pid: str) -> None:
        if self._target_participant == pid:
            self._target_participant = ""
            self._output.set_target_participant("")

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()
