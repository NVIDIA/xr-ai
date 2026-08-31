# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeviceIOHub transport for Pipecat.

Bridges ``ProcessorEndpoint`` (ZMQ IPC) to Pipecat's frame pipeline.

Input  — float32 audio chunks from the hub at any sample rate, resampled
         to 16 kHz int16 ``InputAudioRawFrame`` for the STT processor.
Output — int16 PCM frames written by the TTS processor are converted back
         to float32 ``AudioChunk``s and pushed via ``send_return_audio``.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

import numpy as np
from loguru import logger
from scipy.signal import resample_poly
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from xr_ai_hub import (
    AudioChunk,
    DataMessage,
    ParticipantEvent,
    ProcessorEndpoint,
    Subscribe,
)
from xr_ai_hub._capture import CAPTURE_TTS_TOPIC

from ._audio import float32_to_int16, int16_to_float32
from ._capture_frames import _CaptureTtsCaptionFrame
from ._frames import ParticipantJoinedFrame, ParticipantLeftFrame

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

SAMPLE_RATE            = 16_000
NUM_CHANNELS           = 1
TTS_NATIVE_SAMPLE_RATE = 22_050

# Keep a small amount of audio queued ahead of LiveKit playout. An exact-rate
# producer has no reserve for normal event-loop or IPC latency, so every late
# wakeup becomes an audible underrun. The hub still enforces its independent
# hard queue bound for faulty producers.
_RETURN_AUDIO_TARGET_BUFFER_S = 0.12

# The return track is created lazily by the LiveKit connector when it receives
# the first audio chunk. Put silence at the head of that new track so its
# publication/subscription handshake cannot consume the start of real speech.
_RETURN_AUDIO_PREROLL_S = 0.32
_RETURN_AUDIO_CHUNK_S = 0.04


def _monotonic_s() -> float:
    return asyncio.get_running_loop().time()


async def _sleep_s(delay_s: float) -> None:
    await asyncio.sleep(delay_s)


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

class DeviceIOHubInputTransport(BaseInputTransport):
    """Hub → Pipecat: float32 hub audio → 16 kHz int16 pipecat frames."""

    def __init__(
        self,
        ep: ProcessorEndpoint,
        params: TransportParams,
        *,
        return_audio_primer: Callable[[str], None] | None = None,
        started_event: asyncio.Event | None = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._ep = ep
        self._ep_task: asyncio.Task | None = None
        self._started = False
        self._started_event = started_event
        self._return_audio_primer = return_audio_primer
        # ProcessorEndpoint detaches lifecycle callbacks. Capture their order
        # synchronously, before detachment, so later lifecycle work cannot
        # overtake a participant's join frame.
        self._participant_event_tails: dict[str, asyncio.Future[None]] = {}
        self._ep.on_audio(self._on_hub_audio)
        self._ep.on_participant(self._on_hub_participant)

    async def start(self, frame: StartFrame):
        if self._started_event:
            self._started_event.clear()
        await super().start(frame)
        self._started = True
        self._ep_task = asyncio.create_task(self._ep.run(), name="ep-run")
        await self._ep.wait_until_running()
        if self._started_event:
            self._started_event.set()
        logger.info("DeviceIOHubInputTransport started")

    async def stop(self, frame: EndFrame):
        self._started = False
        if self._started_event:
            self._started_event.clear()
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
        if self._started_event:
            self._started_event.clear()
        self._ep.stop()
        if self._ep_task:
            self._ep_task.cancel()
        await super().cancel(frame)

    async def _on_hub_audio(self, chunk: AudioChunk) -> None:
        if not self._started:
            return
        pcm_int16 = _hub_pcm_to_mono_16k(
            float32_to_int16(chunk.data), chunk.channels, chunk.sample_rate,
        )
        frame = InputAudioRawFrame(
            audio=pcm_int16,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        # pipecat's ``transport_source`` is the standard "which input
        # track did this come from" hook — set it to the hub-side
        # participant id so downstream processors (VadStt, assistant, the
        # output transport's return_data / return_audio routing) can
        # address the right participant. Without this, every downstream
        # send falls back to the empty string and the hub drops the
        # message.
        frame.transport_source = chunk.participant_id
        # Carry the hub's capture time forward on pipecat's standard presentation
        # timestamp (nanoseconds). This is what anchors a turn to when the
        # participant actually spoke; stamping wall-clock after STT instead would
        # bake in VAD hangover plus transcription latency, which then persists
        # into transcripts and time-relative recorded-frame lookups.
        frame.pts = chunk.pts_us * 1_000
        await self.push_frame(frame)

    def _on_hub_participant(self, event: ParticipantEvent) -> Awaitable[None]:
        """Serialize one participant's detached lifecycle callbacks."""
        pid = event.participant_id
        predecessor = self._participant_event_tails.get(pid)
        completed = asyncio.get_running_loop().create_future()
        self._participant_event_tails[pid] = completed

        async def deliver() -> None:
            try:
                if predecessor is not None:
                    await asyncio.shield(predecessor)
                await self._deliver_hub_participant(event)
            finally:
                if not completed.done():
                    completed.set_result(None)
                if self._participant_event_tails.get(pid) is completed:
                    self._participant_event_tails.pop(pid, None)

        return deliver()

    async def _deliver_hub_participant(self, event: ParticipantEvent) -> None:
        """Translate an ordered hub event into a pipecat lifecycle frame.

        The hub publishes one event per LiveKit join/leave; downstream
        processors (``VoiceGateProcessor`` greeting hook,
        the assistant processor) consume the resulting
        ``ParticipantJoinedFrame`` / ``ParticipantLeftFrame``. Without
        this bridge the gate never greets and the assistant never steers the
        output transport at a participant, so every TTS chunk is dropped
        by ``DeviceIOHubOutputTransport.write_audio_frame``.

        Same ``_started`` guard as ``_on_hub_audio``: a late event after
        teardown is a no-op rather than racing the pipeline shutdown.
        """
        if not self._started:
            return
        if event.joined:
            if self._return_audio_primer is not None:
                try:
                    self._return_audio_primer(event.participant_id)
                except Exception:
                    # Track preparation improves first-response quality but
                    # must never suppress participant lifecycle delivery.
                    logger.opt(exception=True).warning(
                        "return-audio pre-roll start failed pid={!r}",
                        event.participant_id,
                    )
            await self.push_frame(
                ParticipantJoinedFrame(participant_id=event.participant_id),
            )
        else:
            await self.push_frame(
                ParticipantLeftFrame(participant_id=event.participant_id),
            )

# ── Output ────────────────────────────────────────────────────────────────────

class DeviceIOHubOutputTransport(BaseOutputTransport):
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
        # Pace each participant independently before IPC. DeviceIOHub keeps a
        # hard safety bound, but normal long replies should arrive near playback
        # rate instead of looking like an unbounded producer burst.
        self._return_audio_deadline_s: dict[str, float] = {}
        self._return_audio_locks: dict[str, asyncio.Lock] = {}
        self._return_audio_preroll_tasks: dict[str, asyncio.Task[None]] = {}
        self._pending_capture_captions: dict[str, deque[str]] = {}
        self._capture_caption_tasks: set[asyncio.Task[None]] = set()
        self._capture_caption_lock = asyncio.Lock()

    @property
    def target_participant(self) -> str:
        return self._target_participant

    def set_target_participant(self, pid: str) -> None:
        # Retained as a fallback for frames that reach the output with no
        # ``transport_destination`` (routed through the default ``None``
        # sender). Per-participant routing now keys on the frame's own pid
        # (see ``write_audio_frame``), so this no longer has to be a single
        # room-wide target.
        logger.info("fallback target participant set pid={!r}", pid)
        self._target_participant = pid
        self._missing_target_warned = False

    def start_return_audio_preroll(self, pid: str) -> None:
        """Start one ordered 320 ms return-track pre-roll for ``pid``."""
        current = self._return_audio_preroll_tasks.get(pid)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._run_return_audio_preroll(pid),
            name=f"return-audio-preroll-{pid}",
        )
        self._return_audio_preroll_tasks[pid] = task

    async def _run_return_audio_preroll(self, pid: str) -> None:
        """Pace silence at audio rate so a small hub queue cannot drop it."""
        samples = round(TTS_NATIVE_SAMPLE_RATE * _RETURN_AUDIO_CHUNK_S)
        data = bytes(samples * NUM_CHANNELS * np.dtype(np.float32).itemsize)
        chunks = round(_RETURN_AUDIO_PREROLL_S / _RETURN_AUDIO_CHUNK_S)
        pts_us = time.time_ns() // 1_000
        lock = self._return_audio_locks.setdefault(pid, asyncio.Lock())
        task = asyncio.current_task()
        try:
            async with lock:
                for index in range(chunks):
                    await self._send_paced_return_audio(
                        AudioChunk(
                            pts_us=(
                                pts_us
                                + round(
                                    index * _RETURN_AUDIO_CHUNK_S * 1_000_000,
                                )
                            ),
                            sample_rate=TTS_NATIVE_SAMPLE_RATE,
                            channels=NUM_CHANNELS,
                            samples=samples,
                            data=data,
                            participant_id=pid,
                            track_id="tts",
                        ),
                        target_buffer_s=_RETURN_AUDIO_CHUNK_S,
                    )
            logger.info(
                "return-audio pre-roll queued pid={!r} duration_ms={:.0f}",
                pid,
                _RETURN_AUDIO_PREROLL_S * 1_000,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Track preparation improves first-response quality but failure
            # cannot take down the detached participant callback or TTS sender.
            logger.opt(exception=True).warning(
                "return-audio pre-roll failed pid={!r}", pid,
            )
        finally:
            if self._return_audio_preroll_tasks.get(pid) is task:
                self._return_audio_preroll_tasks.pop(pid, None)

    async def _wait_return_audio_preroll(self, pid: str) -> bool:
        """Keep real speech ordered behind an active pre-roll.

        Return ``False`` when participant cleanup cancelled the pre-roll so
        the waiting audio frame is dropped without cancelling Pipecat's
        long-lived media-sender task. Cancellation of the caller itself must
        still propagate through the shield.
        """
        task = self._return_audio_preroll_tasks.get(pid)
        if task is None:
            return True
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            waiter = asyncio.current_task()
            if waiter is not None and waiter.cancelling():
                raise
            if task.cancelled():
                return False
            raise
        return True

    async def _cancel_return_audio_preroll(self, pid: str) -> None:
        task = self._return_audio_preroll_tasks.pop(pid, None)
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Expected after cancelling this participant's pre-roll above.
            pass

    async def _cancel_all_return_audio_prerolls(self) -> None:
        tasks = list(self._return_audio_preroll_tasks.values())
        self._return_audio_preroll_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_paced_return_audio(
        self,
        chunk: AudioChunk,
        *,
        target_buffer_s: float,
    ) -> None:
        """Send one lock-protected chunk with a bounded downstream reserve."""
        duration_s = chunk.samples / chunk.sample_rate
        now_s = _monotonic_s()
        buffered_until_s = max(
            self._return_audio_deadline_s.get(chunk.participant_id, now_s),
            now_s,
        )
        delay_s = max(
            0.0,
            buffered_until_s + duration_s - now_s - target_buffer_s,
        )
        if delay_s > 1e-9:
            await _sleep_s(delay_s)
        await self._ep.send_return_audio(chunk)
        sent_s = _monotonic_s()
        self._return_audio_deadline_s[chunk.participant_id] = (
            max(buffered_until_s, sent_s) + duration_s
        )

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
        # pid has somewhere to route, and release it on leave so a long-lived
        # hub with join/leave churn does not retain idle senders until shutdown.
        if isinstance(frame, ParticipantJoinedFrame):
            await self._ensure_destination(frame.participant_id)
        elif isinstance(frame, ParticipantLeftFrame):
            await self._release_destination(frame.participant_id)

    async def _release_destination(self, pid: str) -> None:
        """Tear down a departed participant's ``MediaSender``."""
        await self._cancel_return_audio_preroll(pid)
        lock = self._return_audio_locks.setdefault(pid, asyncio.Lock())
        async with lock:
            sender = self._media_senders.pop(pid, None)
            if sender is not None:
                try:
                    await sender.cancel(CancelFrame())
                except Exception:
                    logger.opt(exception=True).debug(
                        "media sender cancel failed pid={!r}", pid,
                    )
            self._return_audio_deadline_s.pop(pid, None)
            self._pending_capture_captions.pop(pid, None)
        if self._return_audio_locks.get(pid) is lock:
            self._return_audio_locks.pop(pid, None)
        if self._target_participant == pid:
            self._target_participant = ""

    async def stop(self, frame: EndFrame):
        await self._cancel_all_return_audio_prerolls()
        self._return_audio_deadline_s.clear()
        self._return_audio_locks.clear()
        await super().stop(frame)
        await self._finish_capture_caption_tasks()

    async def cancel(self, frame: CancelFrame):
        await self._cancel_all_return_audio_prerolls()
        self._return_audio_deadline_s.clear()
        self._return_audio_locks.clear()
        self._pending_capture_captions.clear()
        await super().cancel(frame)
        await self._cancel_capture_caption_tasks()

    async def _handle_frame(self, frame: Frame) -> None:
        """Funnel every output frame through the default media sender.

        Pipecat's ``BaseOutputTransport._handle_frame`` routes a frame to
        ``_media_senders[frame.transport_destination]`` and drops it
        (with a warning) when the destination is not registered. Only the
        default ``None`` sender is registered by ``set_transport_ready``;
        upstream processors (``VoiceGateProcessor``,
        ``StreamingTtsProcessor``) tag outbound audio with
        ``transport_destination = pid`` so the hub knows which
        participant to send it back to.

        Per-participant routing: each pid has its own ``MediaSender``
        (created on join, or lazily here). The frame keeps its
        ``transport_destination = pid`` so the router delivers it to that
        participant's sender, which stamps the pid back onto the chunk for
        ``write_audio_frame``. Only frames with no pid (or arriving before
        the sender could be created) fall back to the default ``None``
        sender + ``_target_participant``.
        """
        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                sender = self._media_senders.get(pid)
                if sender is not None:
                    await sender.handle_interruptions(frame)
                lock = self._return_audio_locks.setdefault(pid, asyncio.Lock())
                async with lock:
                    self._return_audio_deadline_s.pop(pid, None)
                    self._pending_capture_captions.pop(pid, None)
            else:
                for sender in list(self._media_senders.values()):
                    await sender.handle_interruptions(frame)
                for reset_pid, lock in list(self._return_audio_locks.items()):
                    async with lock:
                        self._return_audio_deadline_s.pop(reset_pid, None)
                self._return_audio_deadline_s.clear()
                self._pending_capture_captions.clear()
            return

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

        The upstream hook is ``write_audio_frame`` (per-frame, returns
        bool), NOT ``write_raw_audio_frames``.
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
        num_samples = len(frame.audio) // (2 * frame.num_channels)
        if not await self._wait_return_audio_preroll(pid):
            return False
        lock = self._return_audio_locks.setdefault(pid, asyncio.Lock())
        async with lock:
            # Send immediately while the estimated downstream reserve is below
            # the target. Once full, wait only long enough to make room for this
            # frame. A scheduler/IPC stall drains the estimate and therefore
            # causes subsequent frames to catch up instead of preserving a gap.
            chunk = AudioChunk(
                pts_us=time.time_ns() // 1_000,
                sample_rate=frame.sample_rate,
                channels=frame.num_channels,
                samples=num_samples,
                data=int16_to_float32(frame.audio),
                participant_id=pid,
                track_id="tts",
            )
            await self._send_paced_return_audio(
                chunk,
                target_buffer_s=_RETURN_AUDIO_TARGET_BUFFER_S,
            )
            self._publish_next_capture_caption(pid, chunk.pts_us)
        return True

    async def write_transport_frame(self, frame: Frame) -> None:
        if isinstance(frame, _CaptureTtsCaptionFrame):
            pid = frame.transport_destination or self._target_participant
            if pid:
                self._pending_capture_captions.setdefault(pid, deque()).append(
                    frame.text
                )
            return
        await super().write_transport_frame(frame)

    def _publish_next_capture_caption(self, pid: str, pts_us: int) -> None:
        captions = self._pending_capture_captions.get(pid)
        if not captions:
            return
        text = captions.popleft()
        if not captions:
            self._pending_capture_captions.pop(pid, None)
        task = asyncio.create_task(
            self._send_capture_caption(pid, pts_us, text),
            name=f"capture-tts-caption-{pid}",
        )
        self._capture_caption_tasks.add(task)
        task.add_done_callback(self._capture_caption_tasks.discard)

    async def _send_capture_caption(
        self,
        pid: str,
        pts_us: int,
        text: str,
    ) -> None:
        try:
            async with self._capture_caption_lock:
                await self._ep.send_return_data(DataMessage(
                    participant_id=pid,
                    topic=CAPTURE_TTS_TOPIC,
                    pts_us=pts_us,
                    data=text.encode(),
                ))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).debug(
                "capture TTS caption failed pid={!r}", pid,
            )

    async def _finish_capture_caption_tasks(self) -> None:
        tasks = tuple(self._capture_caption_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cancel_capture_caption_tasks(self) -> None:
        tasks = tuple(self._capture_caption_tasks)
        self._capture_caption_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ── Transport wrapper ─────────────────────────────────────────────────────────

class HubVoiceTransport(BaseTransport):
    """Own the hub endpoint and bidirectional voice media transport."""

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
            announces_readiness=True,
        )

        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_in_channels=NUM_CHANNELS,
            audio_out_enabled=True,
            audio_out_sample_rate=TTS_NATIVE_SAMPLE_RATE,
            audio_out_channels=NUM_CHANNELS,
        )

        self._input_started = asyncio.Event()
        self._output = DeviceIOHubOutputTransport(self._ep, params, name=self._output_name)
        self._input = DeviceIOHubInputTransport(
            self._ep,
            params,
            name=self._input_name,
            return_audio_primer=self._output.start_return_audio_preroll,
            started_event=self._input_started,
        )

    def input(self) -> DeviceIOHubInputTransport:
        """Return the Pipecat input transport backed by DeviceIOHub."""

        return self._input

    def output(self) -> DeviceIOHubOutputTransport:
        """Return the Pipecat output transport backed by DeviceIOHub."""

        return self._output

    @property
    def endpoint(self) -> ProcessorEndpoint:
        """Return the owned hub processor endpoint."""

        return self._ep

    async def wait_until_started(self) -> None:
        """Wait until the input transport has started its hub IPC receiver."""
        await self._input_started.wait()

    async def send_return_data(self, msg: DataMessage) -> None:
        """Send participant-routed data through the hub."""

        await self._ep.send_return_data(msg)

    @property
    def target_participant(self) -> str:
        """Return the participant currently selected for output."""

        return self._output.target_participant

    def set_target_participant(self, pid: str) -> None:
        """Select the participant that receives subsequent output."""

        self._output.set_target_participant(pid)

    def cleanup_participant(self, pid: str) -> None:
        """Clear output routing when the selected participant leaves."""

        if self._output.target_participant == pid:
            self._output.set_target_participant("")

    def shutdown(self) -> None:
        """Stop and close the owned hub endpoint."""

        self._ep.stop()
        self._ep.close()
