# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private speaker-enrolled alternative to the VAD and batch-STT processor."""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from xr_ai_voicegate._phrases import STOP_RE
from xr_ai_voicegate._speaker import _SpeakerConfig

from .._frames import ParticipantJoinedFrame, ParticipantLeftFrame, _SpeakerEnrollmentFrame, _SpeakerTranscriptionFrame
from .._speaker_client import _SpeakerClient


class _SpeakerSttProcessor(FrameProcessor):
    def __init__(self, *, cfg: _SpeakerConfig, on_partial_transcript=None, on_final_transcript=None) -> None:
        super().__init__()
        self._cfg = cfg
        self._client = _SpeakerClient(cfg)
        self._on_partial = on_partial_transcript
        self._on_final = on_final_transcript
        self._enrolled: set[str] = set()
        self._interrupted: set[str] = set()
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._queued_bytes: dict[str, int] = {}
        self._retry_after: dict[str, float] = {}
        self._closed = False
        self._participants: set[str] = set()
        self._tracks: dict[str, str] = {}

    async def cleanup(self) -> None:
        await self._shutdown()
        await super().cleanup()

    async def _shutdown(self) -> None:
        self._closed = True
        self._participants.clear()
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)
        await self._client._close()
        self._enrolled.clear()
        self._interrupted.clear()
        self._retry_after.clear()
        self._tracks.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._shutdown()
        elif isinstance(frame, ParticipantJoinedFrame):
            pid = frame.participant_id
            await self._cancel_worker(pid)
            self._retry_after.pop(pid, None)
            self._tracks.pop(pid, None)
            if not self._closed:
                self._participants.add(pid)
        elif isinstance(frame, ParticipantLeftFrame):
            pid = frame.participant_id
            self._participants.discard(pid)
            await self._cancel_worker(pid)
            self._retry_after.pop(pid, None)
            self._enrolled.discard(pid)
            self._interrupted.discard(pid)
            self._tracks.pop(pid, None)
        elif isinstance(frame, InputAudioRawFrame):
            pid = frame.transport_source
            if not pid or self._closed or pid not in self._participants:
                return
            if frame.sample_rate != 16000 or frame.num_channels != 1:
                raise ValueError("speaker ASR requires 16 kHz mono PCM")
            if len(frame.audio) > 32000 or len(frame.audio) % 2:
                raise ValueError("speaker ASR expects at most one second of signed 16-bit PCM")
            track = getattr(frame, "track_id", "")
            if track and self._tracks.get(pid, track) != track:
                await self._cancel_worker(pid)
                if self._closed or pid not in self._participants:
                    return
                await self._reset(pid, retry=False)
            if track:
                self._tracks[pid] = track
            if time.monotonic() < self._retry_after.get(pid, 0):
                return
            pts_us = frame.pts // 1_000 if frame.pts is not None else time.time_ns() // 1_000
            queue = self._queues.get(pid)
            if queue is None:
                # The hub emits 10 ms frames: 200 frames match the two-second
                # PCM limit below, rather than imposing a one-second limit.
                queue = asyncio.Queue(maxsize=200)
                self._queues[pid] = queue
                self._queued_bytes[pid] = 0
                self._tasks[pid] = asyncio.create_task(self._run(pid, queue), name=f"speaker-stt-{pid}")
            # Bound both frame count and duration. Replaying delayed control
            # phrases after overload could enroll or release the wrong session.
            if queue.full() or self._queued_bytes[pid] + len(frame.audio) > 64000:
                logger.warning("speaker ASR overloaded; enrollment reset pid={!r}", pid)
                await self._cancel_worker(pid)
                await self._reset(pid)
                return
            self._queued_bytes[pid] += len(frame.audio)
            queue.put_nowait((frame.audio, pts_us))
            return
        await self.push_frame(frame, direction)

    async def _run(self, pid: str, queue: asyncio.Queue) -> None:
        task = asyncio.current_task()
        try:
            while True:
                audio, pts_us = await queue.get()
                self._queued_bytes[pid] -= len(audio)
                for event in await self._client._feed(pid, audio, pts_us):
                    await self._event(pid, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("speaker ASR unavailable; enrollment reset pid={!r}", pid)
            await self._reset(pid)
        finally:
            try:
                # Best-effort cleanup must not hold up disconnect or overload
                # handling when the inference service is unavailable.
                async with asyncio.timeout(min(0.25, self._cfg.timeout_s)):
                    await self._client._forget(pid)
            except Exception:
                logger.warning("speaker session cleanup failed pid={!r}", pid)
            finally:
                if self._tasks.get(pid) is task:
                    self._tasks.pop(pid, None)
                if self._queues.get(pid) is queue:
                    self._queues.pop(pid, None)
                    self._queued_bytes.pop(pid, None)

    async def _cancel_worker(self, pid: str) -> None:
        task = self._tasks.get(pid)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _reset(self, pid: str, *, retry: bool = True) -> None:
        if retry:
            self._retry_after[pid] = time.monotonic() + 2.0
        else:
            self._retry_after.pop(pid, None)
        self._enrolled.discard(pid)
        self._interrupted.discard(pid)
        await self.push_frame(_SpeakerEnrollmentFrame(pid, "reset"))

    async def _event(self, pid: str, event: dict) -> None:
        kind = event["kind"]
        if kind in {"enrolled", "released", "reset"}:
            if kind == "enrolled":
                self._enrolled.add(pid)
            else:
                self._enrolled.discard(pid)
            self._interrupted.discard(pid)
            await self.push_frame(_SpeakerEnrollmentFrame(pid, kind))
            return
        if pid not in self._enrolled:
            return
        if kind == "speech_start":
            self._interrupted.discard(pid)
            frame = UserStartedSpeakingFrame()
        elif kind == "speech_stop":
            frame = UserStoppedSpeakingFrame()
        elif kind == "partial":
            text = event["text"]
            if self._cfg._could_be_control(text):
                return
            stop = bool(STOP_RE.match(text))
            if self._on_partial is not None:
                stop = await self._on_partial(pid, text) or stop
            if not stop or pid in self._interrupted:
                return
            self._interrupted.add(pid)
            frame = InterruptionFrame()
        elif kind in {"transcript", "control_pending"}:
            text = event["text"]
            pts_us = event["pts_us"]
            if kind == "transcript" and self._on_final is not None:
                await self._on_final(pid, text, pts_us)
            frame = _SpeakerTranscriptionFrame(
                text=text, user_id=pid, timestamp=str(pts_us), speaker_id=event.get("speaker_id"),
                _control_pending=kind == "control_pending",
            )
            frame.pts = pts_us * 1_000
        else:
            raise ValueError(f"unknown speaker event: {kind}")
        frame.transport_source = pid
        await self.push_frame(frame)
