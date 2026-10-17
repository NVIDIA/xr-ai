# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture and save the evidence from each participant's procedure demonstration.

CaptureAgent in agent.py starts and stops a capture session for each
demonstration. While a session is active, this module samples camera images,
obtains visual captions, and saves the narration supplied by the agent.
When capture ends, it finishes saving the evidence packet and its summary
for later review and guide creation.

Caption request construction and response parsing live in captioning.py;
spoken command handling remains in agent.py.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nemo_relay
from loguru import logger
from xr_ai_hub import FrameUnavailable
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import MultiImageQueryRequest, MultiImageQueryTool

from .captioning import caption_query, parse_caption

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class _Frame:
    frame_id: int
    timestamp_us: int
    path: str
    sequence: int
    width: int
    height: int


@dataclass(slots=True)
class _Phase:
    name: str
    started_at_us: int
    ended_at_us: int
    caption_count: int
    summary: str
    latest_delta: str
    evidence_frame_id: int
    evidence_path: str


@dataclass(slots=True)
class _Activity:
    name: str
    started_at_us: int
    ended_at_us: int
    caption_count: int
    phases: list[_Phase] = field(default_factory=list)


@dataclass(slots=True)
class _Session:
    participant_id: str
    session_id: str
    directory: Path
    started_at: str
    started_at_us: int
    frame_count: int = 0
    transcript_count: int = 0
    narration_status: str = "pending"
    caption_count: int = 0
    last_frame_key: tuple[str, int, int] | None = None
    last_captioned_frame: _Frame | None = None
    latest_frame: _Frame | None = None
    previous_caption: dict[str, str] | None = None
    activities: list[_Activity] = field(default_factory=list)
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    pending_writes: set[asyncio.Task[Any]] = field(default_factory=set)
    frame_ready: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: bool = True
    status: str = "recording"
    ended_at: str | None = None
    start_done: asyncio.Event = field(default_factory=asyncio.Event)
    stop_command: dict[str, Any] | None = None


def _now_us() -> int:
    return time.time_ns() // 1_000


def _iso(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).isoformat()


def _safe(value: str) -> str:
    return _SAFE.sub("-", value).strip("-._") or "participant"


def _session_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _clock(timestamp_us: int, started_at_us: int) -> str:
    elapsed = max(0, timestamp_us - started_at_us) // 1_000_000
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _Capture:
    """Collect frames, captions, and narration into per-participant evidence packets.

    CaptureAgent starts and stops this private helper. It owns the background
    sampling and captioning tasks, saved evidence, and derived packet summaries.

    Callers serialize control and narration operations per participant. Each
    session has one sampler and one captioner; locks coordinate these tasks
    with finalization, not arbitrary concurrent frame producers.

    Sampling and caption tasks can outlive a cancelled caller. Finalization
    retains those tasks and drains filesystem writes before sealing the packet.
    """

    def __init__(
        self,
        *,
        sessions_dir: Path,
        current_frame: CurrentFrameTool,
        images: ImageRegistry,
        query_images: MultiImageQueryTool,
        capture_fps: float,
        caption_interval_s: float,
    ) -> None:
        self._sessions_dir = sessions_dir
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._current_frame = current_frame
        self._images = images
        self._query_images = query_images
        self._capture_period_s = 1.0 / capture_fps
        self._capture_fps = capture_fps
        self._caption_interval_s = caption_interval_s
        self._sessions: dict[str, _Session] = {}
        self._finalizers: dict[str, asyncio.Task[None]] = {}
        self._sessions_lock = asyncio.Lock()
        self._stopped = False

    def is_capturing(self, participant_id: str) -> bool:
        """Report whether this participant has an active capture session."""
        state = self._sessions.get(participant_id)
        return state is not None and state.active

    async def start_capture(
        self, participant_id: str, *, cancelled: asyncio.Event | None = None, timestamp_us: int | None = None,
    ) -> None:
        """Start a fresh packet unless this participant is already capturing."""
        async with self._sessions_lock:
            if participant_id in self._sessions or self._stopped or (cancelled is not None and cancelled.is_set()):
                return
            now_us = _now_us() if timestamp_us is None else timestamp_us
            session_id = f"{_session_stamp()}-{_safe(participant_id)[:48]}"
            directory = self._sessions_dir / session_id
            state = _Session(
                participant_id=participant_id,
                session_id=session_id,
                directory=directory,
                started_at=_iso(now_us),
                started_at_us=now_us,
            )
            self._sessions[participant_id] = state
        try:
            # Register before starting filesystem work so shutdown can drain it.
            await self._write(state, lambda: (directory / "frames").mkdir(parents=True, exist_ok=False))
            if self._stopped or (cancelled is not None and cancelled.is_set()):
                state.status = "incomplete"
                return
            await self._write(state, _atomic_text, state.directory / "transcript.jsonl", "")
            async with state.lock:
                await self._write_views(state)
            if self._stopped or (cancelled is not None and cancelled.is_set()):
                state.status = "incomplete"
                return
            state.tasks.extend(
                (
                    asyncio.create_task(
                        self._capture_loop(state),
                        name=f"procedure-capture-guide:{participant_id}",
                        context=nemo_relay.fork_asyncio_context(),
                    ),
                    asyncio.create_task(
                        self._caption_loop(state),
                        name=f"task-caption:{participant_id}",
                        context=nemo_relay.fork_asyncio_context(),
                    ),
                )
            )
            logger.info("capture started pid={!r} session={}", participant_id, directory)
        except BaseException:
            async with state.lock:
                state.active = False
                state.status = "incomplete"
            raise
        finally:
            # Finalization may run concurrently with startup, but may not seal
            # the packet while startup can still add tasks or write metadata.
            state.start_done.set()
            if state.status == "incomplete":
                await self._close(participant_id, status="incomplete")

    async def capture_transcript(self, participant_id: str, text: str, timestamp_us: int) -> None:
        """Save a final VoiceAgent transcript inside the active task boundary."""
        state = self._sessions.get(participant_id)
        if state is None or not text.strip():
            return
        async with state.lock:
            if not state.active:
                return
            try:
                await self._write(
                    state, _append_jsonl, state.directory / "transcript.jsonl",
                    {"transcript_id": state.transcript_count + 1, "timestamp_us": timestamp_us,
                     "timestamp": _iso(timestamp_us), "text": text.strip()},
                )
            except Exception:
                state.narration_status = "failed"
                state.status = "incomplete"
                raise
            state.transcript_count += 1

    async def finish_capture(
        self, participant_id: str, *, stop_command: dict[str, Any] | None = None,
    ) -> None:
        """Finish a participant's capture and save its final packet and summary.

        Preserve the accepted stop command when supplied. Disconnect and worker
        shutdown use the same finalization path so pending evidence is drained.
        """
        if stop_command is not None and (state := self._sessions.get(participant_id)) is not None:
            state.stop_command = stop_command
        await self._close(participant_id, status="complete")

    async def stop(self) -> None:
        """Finalize every open packet before worker shutdown."""

        self._stopped = True
        async with self._sessions_lock:
            participants = tuple(self._sessions.keys() | self._finalizers.keys())
        results = await asyncio.gather(
            *(self._close(pid, status="complete") for pid in participants),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("capture finalization failed", errors)
        self._finalizers.clear()

    def release(self, participant_id: str) -> None:
        """Release live frame references only when their connection ends."""
        self._current_frame.release(participant_id)

    async def _write(self, state: _Session, function: Callable[..., Any], *args: Any) -> Any:
        # Cancelling to_thread's awaiter cannot stop its filesystem operation.
        # Keep the actual task alive so finalization can drain it before sealing.
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        state.pending_writes.add(task)
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            state.status = "incomplete"
            state.pending_writes.discard(task)
            raise
        else:
            state.pending_writes.discard(task)
            return result

    async def _drain_writes(self, state: _Session) -> None:
        pending = tuple(state.pending_writes)
        results = await asyncio.gather(*pending, return_exceptions=True)
        state.pending_writes.difference_update(pending)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("capture filesystem writes failed", errors)

    async def _capture_loop(self, state: _Session) -> None:
        while True:
            started_at = time.monotonic()
            try:
                await self._capture(state)
            except asyncio.CancelledError:
                raise
            except FrameUnavailable as exc:
                await self._error(state, "frame_unavailable", str(exc))
            except Exception as exc:
                logger.opt(exception=True).warning("frame capture failed pid={!r}", state.participant_id)
                await self._error(state, "capture_error", str(exc))
            elapsed = time.monotonic() - started_at
            await asyncio.sleep(max(0.0, self._capture_period_s - elapsed))

    async def _capture(self, state: _Session) -> None:
        frame = await self._current_frame.execute(CurrentFrameRequest(participant_id=state.participant_id))
        # On-demand images have sequence zero; live sequences reset on republish.
        frame_key = (frame.track_id, frame.timestamp_us, frame.sequence)
        async with state.lock:
            if not state.active or frame_key == state.last_frame_key:
                return
            frame_id = state.frame_count + 1
        image = self._images.resolve(frame.image)
        if not isinstance(image, bytes):
            raise TypeError(f"captured frame resolved to unsupported input: {type(image).__name__}")
        relative = f"frames/frame_{frame_id:06d}_{frame.timestamp_us}.jpg"
        await self._write(state, (state.directory / relative).write_bytes, image)
        frame_entry = {
            "frame_id": frame_id,
            "timestamp_us": frame.timestamp_us,
            "timestamp": _iso(frame.timestamp_us),
            "path": relative,
            "sequence": frame.sequence,
            "width": frame.width,
            "height": frame.height,
        }
        async with state.lock:
            if not state.active:
                return
            state.frame_count = frame_id
            state.last_frame_key = frame_key
            state.latest_frame = _Frame(
                frame_id=frame_id,
                timestamp_us=frame.timestamp_us,
                path=relative,
                sequence=frame.sequence,
                width=frame.width,
                height=frame.height,
            )
            await self._write(
                state,
                _append_jsonl,
                state.directory / "frames" / "index.jsonl",
                frame_entry,
            )
            state.frame_ready.set()

    async def _caption_loop(self, state: _Session) -> None:
        await state.frame_ready.wait()
        while True:
            try:
                await self._caption(state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.opt(exception=True).warning("frame caption failed pid={!r}", state.participant_id)
                await self._error(state, "caption_error", str(exc))
            await asyncio.sleep(self._caption_interval_s)

    async def _caption(self, state: _Session) -> None:
        async with state.lock:
            frame = state.latest_frame
            previous_frame = state.last_captioned_frame
            previous = dict(state.previous_caption) if state.previous_caption else None
        if frame is None or (previous_frame is not None and frame.frame_id == previous_frame.frame_id):
            return
        frames = [previous_frame, frame] if previous_frame is not None else [frame]
        # Register the saved snapshots at query time: live-frame references can
        # be evicted while a caption request is slow or the camera is idle.
        images = [self._images.put(state.directory / item.path, owner=state.session_id) for item in frames]
        query = caption_query(previous)
        result = await self._query_images.execute(MultiImageQueryRequest(images=images, query=query))
        if not result.available:
            raise RuntimeError(result.text)
        try:
            payload = parse_caption(result.text)
        except ValueError as exc:
            raise ValueError(f"Invalid caption: {exc}; raw response: {result.text!r}") from exc
        caption = payload
        generated_at_us = _now_us()
        async with state.lock:
            if not state.active:
                return
            state.caption_count += 1
            caption_entry = {
                "caption_id": state.caption_count,
                "frame_id": frame.frame_id,
                "frame_timestamp_us": frame.timestamp_us,
                "frame_timestamp": _iso(frame.timestamp_us),
                "generated_at_us": generated_at_us,
                "generated_at": _iso(generated_at_us),
                "frame_path": frame.path,
                # This is the actual visual comparison input, not the previous
                # sampled frame: inference can skip samples or fail entirely.
                "delta_reference_frame_id": previous_frame.frame_id if previous_frame else None,
                "delta_reference_timestamp_us": previous_frame.timestamp_us if previous_frame else None,
                **caption,
            }
            await self._write(
                state,
                _append_jsonl,
                state.directory / "captions.jsonl",
                caption_entry,
            )
            state.previous_caption = caption
            state.last_captioned_frame = frame
            self._update_hierarchy(state, caption_entry)

    @staticmethod
    def _update_hierarchy(state: _Session, caption: dict[str, Any]) -> None:
        # Group contiguous matching labels. A phase's summary keeps its latest
        # caption, not an accumulated summary of all captions in that phase.
        timestamp_us = int(caption["frame_timestamp_us"])
        activity_name = str(caption["activity"])
        phase_name = str(caption["phase"])
        if not state.activities or state.activities[-1].name != activity_name:
            state.activities.append(
                _Activity(
                    name=activity_name,
                    started_at_us=timestamp_us,
                    ended_at_us=timestamp_us,
                    caption_count=0,
                )
            )
        activity = state.activities[-1]
        activity.ended_at_us = timestamp_us
        activity.caption_count += 1
        if not activity.phases or activity.phases[-1].name != phase_name:
            activity.phases.append(
                _Phase(
                    name=phase_name,
                    started_at_us=timestamp_us,
                    ended_at_us=timestamp_us,
                    caption_count=0,
                    summary="",
                    latest_delta="",
                    evidence_frame_id=int(caption["frame_id"]),
                    evidence_path=str(caption["frame_path"]),
                )
            )
        phase = activity.phases[-1]
        phase.ended_at_us = timestamp_us
        phase.caption_count += 1
        phase.summary = str(caption["caption"])
        phase.latest_delta = str(caption["delta"])
        phase.evidence_frame_id = int(caption["frame_id"])
        phase.evidence_path = str(caption["frame_path"])

    async def _write_views(self, state: _Session) -> None:
        await self._write_packet(state)
        await self._write(
            state,
            _atomic_text,
            state.directory / "summary.md",
            self._summary(state),
        )

    async def _write_packet(
        self,
        state: _Session,
        *,
        status: str | None = None,
    ) -> None:
        packet = {
            "schema_version": 1,
            "session_id": state.session_id,
            "participant_id": state.participant_id,
            "status": status or state.status,
            "started_at": state.started_at,
            "ended_at": state.ended_at,
            "stop_command": state.stop_command,
            "narration_status": state.narration_status,
            "capture": {
                "target_fps": self._capture_fps,
                "caption_interval_s": self._caption_interval_s,
            },
            "files": {
                "frame_index": "frames/index.jsonl",
                "transcript": "transcript.jsonl",
                "captions": "captions.jsonl",
                "summary": "summary.md",
            },
            "counts": {
                "frames": state.frame_count,
                "transcripts": state.transcript_count,
                "captions": state.caption_count,
            },
            "hierarchy": [asdict(activity) for activity in state.activities],
        }
        await self._write(
            state,
            _atomic_json,
            state.directory / "packet.json",
            packet,
        )

    def _summary(self, state: _Session) -> str:
        lines = [
            f"# Capture {state.session_id}",
            "",
            f"Status: {state.status}  ",
            f"Started: {state.started_at}  ",
            (f"Frames: {state.frame_count} · Transcripts: {state.transcript_count} · Captions: {state.caption_count}"),
            "",
            "## Hierarchical visual summary",
            "",
            "| Level | Time range | Summary | Latest delta | Evidence |",
            "|---|---:|---|---|---|",
        ]
        if not state.activities:
            lines.append("| Activity | 00:00:00 | Waiting for the first visual caption |  |  |")
        for number, activity in enumerate(state.activities, start=1):
            activity_range = (
                f"{_clock(activity.started_at_us, state.started_at_us)}–"
                f"{_clock(activity.ended_at_us, state.started_at_us)}"
            )
            lines.append(
                f"| **Activity {number}** | {activity_range} | "
                f"**{_markdown(activity.name)}** |  | {activity.caption_count} captions |"
            )
            for phase in activity.phases:
                phase_range = (
                    f"{_clock(phase.started_at_us, state.started_at_us)}–"
                    f"{_clock(phase.ended_at_us, state.started_at_us)}"
                )
                evidence = f"[frame {phase.evidence_frame_id}]({phase.evidence_path})"
                lines.append(
                    f"| ↳ Phase | {phase_range} | **{_markdown(phase.name)}:** "
                    f"{_markdown(phase.summary)} | {_markdown(phase.latest_delta)} | "
                    f"{evidence} |"
                )
        lines.extend(
            (
                "",
                (
                    "Frames and captions are source records; narration uses final "
                    "transcripts from the shared voice runtime. `packet.json` "
                    "is the current machine-readable index for coding agents."
                ),
                "",
            )
        )
        return "\n".join(lines)

    async def _error(self, state: _Session, kind: str, message: str) -> None:
        async with state.lock:
            if not state.active:
                return
            try:
                await self._write(
                    state,
                    _append_jsonl,
                    state.directory / "errors.jsonl",
                    {"timestamp_us": _now_us(), "kind": kind, "message": message},
                )
            except Exception:
                logger.exception("capture error log failed pid={!r} kind={}", state.participant_id, kind)

    async def _close(self, participant_id: str, *, status: str) -> None:
        async with self._sessions_lock:
            state = self._sessions.get(participant_id)
            finalizer = self._finalizers.get(participant_id)
            if finalizer is None and state is None:
                return
            if state is not None and (finalizer is None or finalizer.done()):
                finalizer = asyncio.create_task(
                    self._finalize(state, status=status),
                    name=f"task-finalize:{participant_id}",
                    context=nemo_relay.fork_asyncio_context(),
                )
                self._finalizers[participant_id] = finalizer
        assert finalizer is not None
        await asyncio.shield(finalizer)

    async def _finalize(self, state: _Session, *, status: str) -> None:
        await state.start_done.wait()
        participant_id = state.participant_id
        for task in state.tasks:
            task.cancel()
        results = await asyncio.gather(*state.tasks, return_exceptions=True)
        async with state.lock:
            state.active = False
            # STT can deliver a stop after later frames have already been saved.
            # Keep its requested boundary in stop_command; ended_at records when
            # producers have stopped and no further evidence can be admitted.
            state.ended_at = _iso(_now_us())
        if any(
            isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError) for result in results
        ):
            state.status = "incomplete"
        try:
            await self._drain_writes(state)
        except BaseExceptionGroup:
            state.status = "incomplete"
            logger.exception("capture writes failed pid={!r}", participant_id)
        try:
            await self._rebuild(state)
            if state.status != "incomplete":
                state.status = status
            await self._write_views(state)
        except Exception:
            state.status = "incomplete"
            logger.exception("capture finalization failed pid={!r}", participant_id)
            # A failed view must never leave a packet advertised as complete.
            # If storage is still unavailable, preserve the error in process logs.
            try:
                await self._write_views(state)
            except Exception:
                logger.exception("incomplete packet seal failed pid={!r}", participant_id)
        finally:
            self._images.release_owner(state.session_id)
            async with self._sessions_lock:
                if self._sessions.get(participant_id) is state:
                    self._sessions.pop(participant_id)
        logger.info(
            "capture finalized pid={!r} status={} session={}", participant_id, state.status, state.directory,
        )

    async def _rebuild(self, state: _Session) -> None:
        # A cancelled append awaiter may have skipped the in-memory update.
        # Rebuild derived views from the durable log after all writers finish.
        def read_entries(relative: str) -> list[dict[str, Any]]:
            path = state.directory / relative
            if not path.exists():
                return []
            with path.open(encoding="utf-8") as stream:
                return [json.loads(line) for line in stream if line.strip()]

        frames = await asyncio.to_thread(read_entries, "frames/index.jsonl")
        state.frame_count = len(frames)
        captions = await asyncio.to_thread(read_entries, "captions.jsonl")
        state.caption_count = len(captions)
        state.activities.clear()
        for caption in captions:
            self._update_hierarchy(state, caption)
        # Count the durable log after all queued transcript writes have drained.
        transcript_path = state.directory / "transcript.jsonl"
        if not state.tasks and not transcript_path.exists():
            # Startup was cancelled or failed before narration was initialized.
            return

        def count_transcripts() -> int:
            with transcript_path.open(encoding="utf-8") as stream:
                return sum(bool(line.strip()) for line in stream)
        try:
            state.transcript_count = await asyncio.to_thread(count_transcripts)
            if state.narration_status != "failed":
                state.narration_status = "complete"
        except OSError as exc:
            state.narration_status = "failed"
            state.status = "incomplete"
            logger.warning("capture narration unavailable pid={!r}: {}", state.participant_id, exc)
