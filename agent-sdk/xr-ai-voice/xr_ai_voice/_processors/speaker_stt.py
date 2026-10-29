# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private speaker-enrolled alternative to the VAD and batch-STT processor."""

from __future__ import annotations

import asyncio
import time
import uuid

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

from .._frames import (
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
    _SpeakerDiagnosticFrame,
    _SpeakerEnrollmentFrame,
    _SpeakerTranscriptionFrame,
    _SpeakerTurnFrame,
    _TurnResetFrame,
    _TurnTranscriptFrame,
)
from .._speaker_client import _SpeakerClient
from .._turn_policy import TranscriptUpdate


class _SpeakerSttProcessor(FrameProcessor):
    def __init__(self, *, cfg: _SpeakerConfig, on_partial_transcript=None, on_final_transcript=None,
                 transcript_updates: bool = False) -> None:
        super().__init__()
        self._cfg = cfg
        self._client = _SpeakerClient(cfg)
        self._on_partial = on_partial_transcript
        self._on_final = on_final_transcript
        self._enrolled: set[str] = set()
        self._interrupted: set[str] = set()
        # Participants with a started utterance the server has not stopped yet.
        self._utterances: set[str] = set()
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._queued_bytes: dict[str, int] = {}
        self._retry_after: dict[str, float] = {}
        self._closed = False
        self._participants: set[str] = set()
        self._lifecycles: dict[str, object] = {}
        self._tracks: dict[str, str] = {}
        self._transcript_updates = transcript_updates
        self._epochs: dict[str, str] = {}
        self._segments: dict[str, int] = {}
        self._revisions: dict[str, int] = {}
        self._audio_pts: dict[str, int] = {}
        self._onset_pts: dict[str, int] = {}
        self._held_words: dict[str, str] = {}

    async def cleanup(self) -> None:
        await self._shutdown()
        await super().cleanup()

    async def _shutdown(self) -> None:
        self._closed = True
        self._participants.clear()
        self._lifecycles.clear()
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)
        await self._client._close()
        self._enrolled.clear()
        self._interrupted.clear()
        self._utterances.clear()
        self._retry_after.clear()
        self._tracks.clear()
        self._epochs.clear()
        self._segments.clear()
        self._revisions.clear()
        self._audio_pts.clear()
        self._onset_pts.clear()
        self._held_words.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._shutdown()
        elif isinstance(frame, ParticipantJoinedFrame):
            if self._closed:
                return
            pid = frame.participant_id
            replacing = pid in self._lifecycles
            self._lifecycles[pid] = object()
            if not await self._replace_worker(pid, reset=replacing):
                return
            self._retry_after.pop(pid, None)
            self._tracks.pop(pid, None)
            self._forget_updates(pid)
        elif isinstance(frame, ParticipantLeftFrame):
            pid = frame.participant_id
            self._lifecycles.pop(pid, None)
            self._participants.discard(pid)
            await self._cancel_worker(pid)
            self._retry_after.pop(pid, None)
            self._enrolled.discard(pid)
            self._interrupted.discard(pid)
            self._utterances.discard(pid)
            self._tracks.pop(pid, None)
            self._forget_updates(pid)
            self._audio_pts.pop(pid, None)
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
                if not await self._replace_worker(pid):
                    return
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
                await self._replace_worker(pid, retry=True)
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
                self._audio_pts[pid] = pts_us
                self._queued_bytes[pid] -= len(audio)
                for event in await self._client._feed(pid, audio, pts_us):
                    if self._closed or pid not in self._participants or self._tasks.get(pid) is not task:
                        return
                    await self._event(pid, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("speaker ASR unavailable; enrollment reset pid={!r}", pid)
            if not self._closed and pid in self._participants and self._tasks.get(pid) is task:
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

    async def _replace_worker(self, pid: str, *, reset: bool = True, retry: bool = False) -> bool:
        """Revoke the old worker before draining; readmit only its current join."""
        lifecycle = self._lifecycles.get(pid)
        self._participants.discard(pid)
        self._enrolled.discard(pid)
        self._interrupted.discard(pid)
        await self._cancel_worker(pid)
        await self._end_utterance(pid)
        if self._closed or lifecycle is None or self._lifecycles.get(pid) is not lifecycle:
            return False
        if reset:
            await self._reset(pid, retry=retry)
        if self._closed or self._lifecycles.get(pid) is not lifecycle:
            return False
        self._participants.add(pid)
        return True

    async def _reset(self, pid: str, *, retry: bool = True) -> None:
        self._forget_updates(pid)
        if self._transcript_updates:
            await self.push_frame(_TurnResetFrame(pid))
        if retry:
            self._retry_after[pid] = time.monotonic() + 2.0
        else:
            self._retry_after.pop(pid, None)
        self._enrolled.discard(pid)
        self._interrupted.discard(pid)
        await self._end_utterance(pid)
        await self.push_frame(_SpeakerEnrollmentFrame(pid, "reset"))

    async def _end_utterance(self, pid: str) -> None:
        """Stop an utterance whose stop event can no longer arrive.

        Speech events are dropped once enrollment ends, so without this the
        participant would stay speaking downstream and hold back voice updates.
        """

        if pid not in self._utterances:
            return
        self._utterances.discard(pid)
        frame = UserStoppedSpeakingFrame()
        frame.transport_source = pid
        await self.push_frame(frame)

    async def _event(self, pid: str, event: dict) -> None:
        kind = event["kind"]
        if kind in {"one_shot", "wake_ready"}:
            if self._cfg.mode not in {"always_on", "wake_only", "hybrid"} or pid in self._enrolled:
                raise ValueError("one-turn authorization is invalid for the current speaker state")
            speaker, pts = event.get("speaker_id"), event.get("pts_us")
            text = event.get("text") if kind == "one_shot" else ""
            fresh = event.get("fresh_match") if kind == "one_shot" else True
            if (
                type(speaker) is not int or speaker < 0 or type(pts) is not int or pts < 0
                or not isinstance(text, str) or (kind == "one_shot" and not text.strip())
                or not isinstance(fresh, bool)
            ):
                raise ValueError("malformed speaker one-turn authorization")
            if text and self._on_final is not None:
                await self._on_final(pid, text, pts)
            await self.push_frame(_SpeakerTurnFrame(pid, speaker, text, pts, fresh))
            return
        if kind == "diagnostic":
            if self._cfg.diagnostics:
                speaker, pts = event.get("speaker_id"), event.get("pts_us")
                status, text = event.get("status"), event.get("text")
                if (
                    type(speaker) is not int or speaker < 0 or type(pts) is not int or pts < 0
                    or not isinstance(status, str) or not status.strip() or not isinstance(text, str)
                ):
                    logger.warning("malformed speaker diagnostic discarded pid={!r}", pid)
                    return
                await self.push_frame(_SpeakerDiagnosticFrame(
                    participant_id=pid, speaker_id=speaker, status=status, text=text, pts_us=pts,
                ))
            return
        if kind in {"enrolled", "released", "reset"}:
            if self._transcript_updates:
                self._forget_updates(pid)
                if kind == "enrolled":
                    self._epochs[pid] = uuid.uuid4().hex
                    self._segments[pid] = -1
                await self.push_frame(_TurnResetFrame(pid))
            if kind == "enrolled":
                self._enrolled.add(pid)
            else:
                self._enrolled.discard(pid)
                await self._end_utterance(pid)
            self._interrupted.discard(pid)
            await self.push_frame(_SpeakerEnrollmentFrame(pid, kind))
            return
        if pid not in self._enrolled:
            return
        if kind == "speech_start":
            if self._transcript_updates:
                self._segments[pid] = self._segments.get(pid, -1) + 1
                self._revisions[pid] = 0
                self._onset_pts[pid] = self._audio_pts.get(pid, time.time_ns() // 1_000)
            self._interrupted.discard(pid)
            self._utterances.add(pid)
            frame = UserStartedSpeakingFrame()
        elif kind == "speech_stop":
            self._held_words.pop(pid, None)
            self._utterances.discard(pid)
            frame = UserStoppedSpeakingFrame()
        elif kind == "partial":
            text = event["text"]
            if self._cfg._could_be_control(text):
                return
            # The assembled pipeline owns STOP policy, including its opt-out.
            # Only standalone processors fall back to default recognition.
            stop = (
                await self._on_partial(pid, text)
                if self._on_partial is not None else bool(STOP_RE.match(text))
            )
            if not stop or pid in self._interrupted:
                if not stop and self._transcript_updates:
                    await self._emit_update(pid, text, final=False)
                return
            self._interrupted.add(pid)
            frame = InterruptionFrame()
        elif kind in {"transcript", "control_pending"}:
            text = event["text"]
            pts_us = event["pts_us"]
            if kind == "control_pending" and self._transcript_updates:
                self._held_words[pid] = text
            if kind == "transcript" and self._on_final is not None:
                await self._on_final(pid, text, pts_us)
            if kind == "transcript" and self._transcript_updates:
                released = event.get("released", "")
                semantic_text = f"{released} {text}" if released else text
                await self._emit_update(pid, semantic_text, final=True, timestamp_us=pts_us)
            frame = _SpeakerTranscriptionFrame(
                text=text, user_id=pid, timestamp=str(pts_us), speaker_id=event.get("speaker_id"),
                _control_pending=kind == "control_pending", _released=event.get("released", ""),
            )
            frame.pts = pts_us * 1_000
        else:
            raise ValueError(f"unknown speaker event: {kind}")
        frame.transport_source = pid
        await self.push_frame(frame)

    def _forget_updates(self, pid: str) -> None:
        for values in (self._epochs, self._segments, self._revisions, self._onset_pts, self._held_words):
            values.pop(pid, None)

    async def _emit_update(self, pid: str, text: str, *, final: bool, timestamp_us: int | None = None) -> None:
        epoch = self._epochs.get(pid)
        if epoch is None or self._segments.get(pid, -1) < 0:
            return
        if not final and (prefix := self._held_words.get(pid)):
            text = f"{prefix} {text}"
        self._revisions[pid] = self._revisions.get(pid, 0) + 1
        update = TranscriptUpdate(
            participant_id=pid, source_epoch=epoch, segment_id=self._segments[pid],
            revision=self._revisions[pid], text=text,
            timestamp_us=timestamp_us if timestamp_us is not None else self._onset_pts[pid], final=final,
        )
        await self.push_frame(_TurnTranscriptFrame(update))
