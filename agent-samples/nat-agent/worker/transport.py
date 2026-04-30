# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XR-Media-Hub transport for Pipecat.

Bridges ``ProcessorEndpoint`` (ZMQ IPC) to Pipecat's frame pipeline.

Input  — float32 audio chunks from the hub at any sample rate, resampled
         to 16 kHz int16 ``InputAudioRawFrame`` for the STT processor.
Output — int16 PCM frames written by the TTS processor are converted back
         to float32 ``AudioChunk``s and pushed via ``send_return_audio``.

The worker subscribes only to audio + data (no video) — video frames are
the LLM's concern via video-mcp, never the worker's.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from xr_ai_agent import (
    AudioChunk, DataMessage, ProcessorEndpoint, Subscribe,
)

import config as cfg

log = logging.getLogger("nat_agent.transport")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


def _float32_to_int16(data: bytes) -> bytes:
    f32 = np.frombuffer(data, dtype=np.float32)
    return np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def _int16_to_float32(data: bytes) -> bytes:
    i16 = np.frombuffer(data, dtype=np.int16)
    return (i16.astype(np.float32) / 32767.0).tobytes()


# ── Input ────────────────────────────────────────────────────────────────────

class XRMediaHubInputTransport(BaseInputTransport):
    """Hub → Pipecat: float32 hub audio → 16 kHz int16 pipecat frames."""

    def __init__(self, ep: ProcessorEndpoint, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._ep = ep
        self._ep_task: asyncio.Task | None = None
        self._started = False
        self._ep.on_audio(self._on_hub_audio)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._started = True
        self._ep_task = asyncio.create_task(self._ep.run(), name="ep-run")
        log.info("XRMediaHubInputTransport started")

    async def stop(self, frame: EndFrame):
        self._started = False
        self._ep.stop()
        if self._ep_task:
            self._ep_task.cancel()
            try:
                await self._ep_task
            except asyncio.CancelledError:
                pass
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
        pcm_int16 = _float32_to_int16(chunk.data)
        if chunk.sample_rate != cfg.SAMPLE_RATE:
            from scipy.signal import resample_poly
            audio_array = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float64)
            audio_array = resample_poly(
                audio_array, cfg.SAMPLE_RATE, chunk.sample_rate,
            ).astype(np.int16)
            pcm_int16 = audio_array.tobytes()
        await self.push_frame(InputAudioRawFrame(
            audio=pcm_int16,
            sample_rate=cfg.SAMPLE_RATE,
            num_channels=chunk.channels,
        ))


# ── Output ───────────────────────────────────────────────────────────────────

class XRMediaHubOutputTransport(BaseOutputTransport):
    """Pipecat → Hub: int16 TTS frames → float32 ``AudioChunk``s.

    The target participant is set lazily on first inbound traffic so voice
    replies work even if the join event was missed.
    """

    def __init__(self, ep: ProcessorEndpoint, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._ep = ep
        self._target_participant: str = ""

    def set_target_participant(self, pid: str) -> None:
        self._target_participant = pid

    async def start(self, frame: StartFrame):
        await super().start(frame)
        log.info("XRMediaHubOutputTransport started (out_rate=%d)", self.sample_rate)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def write_raw_audio_frames(self, frames: bytes) -> None:
        if not self._target_participant:
            return

        pcm_float32 = _int16_to_float32(frames)
        num_samples = len(frames) // (2 * cfg.NUM_CHANNELS)

        chunk = AudioChunk(
            pts_us=int(time.time() * 1_000_000),
            sample_rate=self.sample_rate,
            channels=cfg.NUM_CHANNELS,
            samples=num_samples,
            data=pcm_float32,
            participant_id=self._target_participant,
            track_id="tts",
        )
        await self._ep.send_return_audio(chunk)


# ── Transport wrapper ────────────────────────────────────────────────────────

class XRMediaHubTransport(BaseTransport):
    """Owns the ProcessorEndpoint + bidirectional Pipecat transports.

    The endpoint is constructed with a filter that drops video frames at the
    ZMQ kernel level — the worker has no use for live frames; video-mcp owns
    that path.
    """

    def __init__(
        self,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)

        self._ep = ProcessorEndpoint(
            sub_addr=_HUB_PUB,
            push_addr=_HUB_PUSH,
            filter=Subscribe.AUDIO | Subscribe.DATA,
        )

        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=cfg.SAMPLE_RATE,
            audio_in_channels=cfg.NUM_CHANNELS,
            audio_out_enabled=True,
            audio_out_sample_rate=cfg.TTS_NATIVE_SAMPLE_RATE,
            audio_out_channels=cfg.NUM_CHANNELS,
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
