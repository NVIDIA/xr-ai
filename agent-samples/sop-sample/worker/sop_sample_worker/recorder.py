# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped frame, transcript, and hierarchical-caption recording."""

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
from xr_ai_hub import FrameUnavailable, ProcessorEndpoint
from xr_ai_runtime import Agent
from xr_ai_tools.capture import CaptureTools
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageReference, ImageRegistry
from xr_ai_tools.tools import ToolSet
from xr_ai_tools.types import EmptyRequest
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_MEDIA_FINALIZE_WARNING_S = 10.0
_MEDIA_START_RETRY_S = 0.1


@dataclass(slots=True)
class _Frame:
    frame_id: int
    timestamp_us: int
    path: str
    sequence: int
    width: int
    height: int
    image: ImageReference


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
    capture_tools: ToolSet
    media_directory: Path
    media_status: str = "start_requested"
    media_manifest: str | None = None
    frame_count: int = 0
    transcript_count: int = 0
    narration_status: str = "pending"
    caption_count: int = 0
    last_sequence: int | None = None
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
    cancel_start: bool = False
    start_done: asyncio.Event = field(default_factory=asyncio.Event)


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


def _export_narration(manifest_path: Path, destination: Path) -> int:
    """Project the capture transcript into the existing SOP narration schema."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = (manifest_path.parent / manifest["transcript"]).resolve()
    if not source.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("capture transcript must be inside its bundle")
    records = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["source"] != "user":
                continue
            text = row["text"].strip()
            if not text:
                continue
            timestamp_us = row["pts_us"]
            if type(timestamp_us) is not int or timestamp_us < 0:
                raise ValueError("capture transcript pts_us must be a non-negative integer")
            records.append(
                json.dumps(
                    {
                        "transcript_id": len(records) + 1,
                        "timestamp_us": timestamp_us,
                        "timestamp": _iso(timestamp_us),
                        "text": text,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    _atomic_text(destination, "".join(records))
    return len(records)


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("caption response did not contain a JSON object")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("caption response was not a JSON object")
    return value


def _clean_field(payload: dict[str, Any], name: str, fallback: str) -> str:
    value = str(payload.get(name, "")).strip()
    return value[:1000] if value else fallback


def _markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _clock(timestamp_us: int, started_at_us: int) -> str:
    elapsed = max(0, timestamp_us - started_at_us) // 1_000_000
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class RecorderAgent(Agent):
    """Capture a durable multimodal packet within camera-controlled boundaries."""

    def __init__(
        self,
        *,
        sessions_dir: Path,
        capture_endpoint: ProcessorEndpoint,
        media_capture_dir: Path,
        current_frame: CurrentFrameTool,
        images: ImageRegistry,
        query_image: ImageQueryTool,
        capture_fps: float,
        caption_interval_s: float,
    ) -> None:
        super().__init__()
        self._sessions_dir = sessions_dir
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._capture_endpoint = capture_endpoint
        self._media_capture_dir = media_capture_dir
        self._current_frame = current_frame
        self._images = images
        self._query_image = query_image
        self._capture_period_s = 1.0 / capture_fps
        self._capture_fps = capture_fps
        self._caption_interval_s = caption_interval_s
        self._sessions: dict[str, _Session] = {}
        self._finalizers: dict[str, asyncio.Task[None]] = {}
        self._sessions_lock = asyncio.Lock()
        self._stopped = False

    def is_recording(self, participant_id: str) -> bool:
        return participant_id in self._sessions

    async def start_recording(self, participant_id: str, *, cancelled: asyncio.Event | None = None) -> None:
        """Start a fresh packet unless this participant is already recording."""
        async with self._sessions_lock:
            if participant_id in self._sessions or self._stopped or (cancelled is not None and cancelled.is_set()):
                return
            now_us = _now_us()
            session_id = f"{_session_stamp()}-{_safe(participant_id)[:48]}"
            directory = self._sessions_dir / session_id
            await asyncio.to_thread(
                (directory / "frames").mkdir,
                parents=True,
                exist_ok=False,
            )
            state = _Session(
                participant_id=participant_id,
                session_id=session_id,
                directory=directory,
                started_at=_iso(now_us),
                started_at_us=now_us,
                capture_tools=CaptureTools(
                    endpoint=self._capture_endpoint,
                    target=session_id,
                    metadata={"sop_session_id": session_id, "sop_packet": str(directory / "packet.json")},
                ).participant_tools(participant_id),
                media_directory=self._media_capture_dir / session_id,
            )
            self._sessions[participant_id] = state
        try:
            if self._stopped or (cancelled is not None and cancelled.is_set()):
                state.media_status = "start_cancelled"
                return
            start = state.capture_tools.get("start_recording")
            assert start is not None
            try:
                await start.execute(EmptyRequest())
            except Exception:
                state.media_status = "start_failed"
                state.status = "failed"
                state.ended_at = _iso(_now_us())
                await self._write_views(state)
                raise
            if not await self._wait_for_media_start(state, cancelled):
                return
            if self._stopped or state.cancel_start or (cancelled is not None and cancelled.is_set()):
                return
            async with state.lock:
                await self._write_views(state)
            state.tasks.extend(
                (
                    asyncio.create_task(
                        self._capture_loop(state),
                        name=f"sop-capture:{participant_id}",
                        context=nemo_relay.fork_asyncio_context(),
                    ),
                    asyncio.create_task(
                        self._caption_loop(state),
                        name=f"sop-caption:{participant_id}",
                        context=nemo_relay.fork_asyncio_context(),
                    ),
                )
            )
            logger.info("recording started pid={!r} session={}", participant_id, directory)
        finally:
            # Finalization may run concurrently with startup, but may not seal
            # the packet while startup can still add tasks or write metadata.
            state.start_done.set()

    @staticmethod
    def _media_started(state: _Session) -> bool:
        for path in state.media_directory.glob("*/events.jsonl"):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    event = json.loads(line)
                    if event.get("kind") == "recording" and event.get("state") == "started":
                        return True
        return False

    async def _wait_for_media_start(self, state: _Session, cancelled: asyncio.Event | None) -> bool:
        # Sending a start is not an acknowledgement: capture may still reject
        # it after publishing the previous manifest, while finishing cleanup.
        # Its flushed start event confirms acceptance in this unique target.
        start = state.capture_tools.get("start_recording")
        assert start is not None
        waiting_since = time.monotonic()
        warned = False
        while not await asyncio.to_thread(self._media_started, state):
            if self._stopped or state.cancel_start or (cancelled is not None and cancelled.is_set()):
                state.media_status = "start_cancelled"
                return False
            if not warned and time.monotonic() - waiting_since >= _MEDIA_FINALIZE_WARNING_S:
                warned = True
                logger.warning("media capture start pending pid={!r}; retrying until accepted", state.participant_id)
            await asyncio.sleep(_MEDIA_START_RETRY_S)
            if self._stopped or state.cancel_start or (cancelled is not None and cancelled.is_set()):
                continue
            # Repeating a start is idempotent while the same session is active.
            await start.execute(EmptyRequest())
        state.media_status = "recording"
        return True

    async def finish_recording(self, participant_id: str) -> None:
        """Finalize the packet using the same path as worker shutdown."""
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
            raise BaseExceptionGroup("recording finalization failed", errors)
        self._finalizers.clear()

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
            raise BaseExceptionGroup("recording filesystem writes failed", errors)

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
        async with state.lock:
            if not state.active or frame.sequence == state.last_sequence:
                return
            frame_id = state.frame_count + 1
        image = self._images.resolve(frame.image)
        if not isinstance(image, bytes):
            raise TypeError(f"captured frame resolved to unsupported input: {type(image).__name__}")
        relative = f"frames/frame_{frame_id:06d}_{frame.timestamp_us}.jpg"
        await self._write(state, (state.directory / relative).write_bytes, image)
        record = {
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
            state.last_sequence = frame.sequence
            state.latest_frame = _Frame(
                frame_id=frame_id,
                timestamp_us=frame.timestamp_us,
                path=relative,
                sequence=frame.sequence,
                width=frame.width,
                height=frame.height,
                image=frame.image,
            )
            await self._write(
                state,
                _append_jsonl,
                state.directory / "frames" / "index.jsonl",
                record,
            )
            state.frame_ready.set()
            await self._write_packet(state)

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
            previous = dict(state.previous_caption) if state.previous_caption else None
        if frame is None:
            return
        query = json.dumps(
            {
                "prior_observation": previous,
                "instruction": "Describe this exact frame and its visible delta.",
            },
            ensure_ascii=False,
        )
        result = await self._query_image.execute(ImageQueryRequest(image=frame.image, query=query))
        if not result.available:
            raise RuntimeError(result.text)
        try:
            payload = _json_object(result.text)
        except (ValueError, json.JSONDecodeError):
            payload = {"caption": result.text}
        caption = {
            "activity": _clean_field(
                payload,
                "activity",
                previous["activity"] if previous else "Unclassified activity",
            ),
            "phase": _clean_field(
                payload,
                "phase",
                previous["phase"] if previous else "Observed work",
            ),
            "caption": _clean_field(
                payload,
                "caption",
                result.text.strip()[:1000] or "Visual observation unavailable",
            ),
            "delta": _clean_field(
                payload,
                "delta",
                "Initial observation" if previous is None else "Delta unavailable",
            ),
        }
        generated_at_us = _now_us()
        async with state.lock:
            if not state.active:
                return
            state.caption_count += 1
            record = {
                "caption_id": state.caption_count,
                "frame_id": frame.frame_id,
                "frame_timestamp_us": frame.timestamp_us,
                "frame_timestamp": _iso(frame.timestamp_us),
                "generated_at_us": generated_at_us,
                "generated_at": _iso(generated_at_us),
                "frame_path": frame.path,
                **caption,
            }
            await self._write(
                state,
                _append_jsonl,
                state.directory / "captions.jsonl",
                record,
            )
            state.previous_caption = caption
            self._update_hierarchy(state, record)
            await self._write_views(state)

    @staticmethod
    def _update_hierarchy(state: _Session, caption: dict[str, Any]) -> None:
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
            "media_capture": {
                "directory": str(state.media_directory),
                "control_status": state.media_status,
                "manifest": state.media_manifest,
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
            f"# Recording {state.session_id}",
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
                    "Frames and captions are source records; narration is derived "
                    "from the shared capture transcript at finalization. `packet.json` "
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
            await self._write(
                state,
                _append_jsonl,
                state.directory / "errors.jsonl",
                {"timestamp_us": _now_us(), "kind": kind, "message": message},
            )

    async def _close(self, participant_id: str, *, status: str) -> None:
        async with self._sessions_lock:
            state = self._sessions.get(participant_id)
            finalizer = self._finalizers.get(participant_id)
            if finalizer is None and state is None:
                return
            if state is not None and (finalizer is None or finalizer.done()):
                finalizer = asyncio.create_task(
                    self._finalize(state, status=status),
                    name=f"sop-finalize:{participant_id}",
                    context=nemo_relay.fork_asyncio_context(),
                )
                self._finalizers[participant_id] = finalizer
        assert finalizer is not None
        await asyncio.shield(finalizer)

    async def _finalize(self, state: _Session, *, status: str) -> None:
        state.cancel_start = True
        await state.start_done.wait()
        accepted = state.media_status == "recording"
        participant_id = state.participant_id
        state.ended_at = _iso(_now_us())
        stop = state.capture_tools.get("stop_recording")
        assert stop is not None
        stop_error: Exception | None = None
        try:
            await stop.execute(EmptyRequest())
            state.media_status = "stop_requested"
        except Exception as exc:
            # The hub may already be closed during shutdown. Preserve the SOP
            # packet; capture also finalizes on departure and service shutdown.
            state.media_status = "stop_failed"
            logger.warning("media capture stop failed pid={!r}: {}", participant_id, exc)
            stop_error = exc
        for task in state.tasks:
            task.cancel()
        if state.tasks:
            await asyncio.gather(*state.tasks, return_exceptions=True)
        await self._drain_writes(state)
        # A cancelled append awaiter may have skipped the in-memory update.
        # Rebuild derived views from the durable log after all writers finish.
        def read_captions() -> list[dict[str, Any]]:
            path = state.directory / "captions.jsonl"
            if not path.exists():
                return []
            with path.open(encoding="utf-8") as stream:
                return [json.loads(line) for line in stream if line.strip()]

        captions = await asyncio.to_thread(read_captions)
        state.caption_count = len(captions)
        state.activities.clear()
        for caption in captions:
            self._update_hierarchy(state, caption)
        if stop_error is not None:
            await self._error(state, "media_capture_stop", str(stop_error))
        # Reconcile acceptance after the last start has settled and stop was
        # sent. An unaccepted target has no manifest to wait for.
        accepted = accepted or await asyncio.to_thread(self._media_started, state)
        if not accepted:
            state.media_status = "start_cancelled"
            await self._error(state, "media_capture_start", "recording ended before capture accepted its start")
        # Departure or service shutdown may finalize capture even if sending
        # stop failed because the voice transport has already closed.
        if accepted:
            await self._wait_for_media(state)
        try:
            if state.media_manifest is None:
                raise ValueError("capture manifest unavailable; narration was not exported")
            state.transcript_count = await self._write(
                state,
                _export_narration,
                Path(state.media_manifest),
                state.directory / "transcript.jsonl",
            )
            state.narration_status = "complete" if state.media_status == "complete" else "incomplete"
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            state.narration_status = "failed"
            logger.warning("narration export failed pid={!r}: {}", participant_id, exc)
            await self._error(state, "narration_export", str(exc))
        if state.narration_status != "complete":
            status = "incomplete"
        async with state.lock:
            state.active = False
            state.status = status
            await self._write_packet(state, status=status)
            await self._write(
                state,
                _atomic_text,
                state.directory / "summary.md",
                self._summary(state),
            )
        self._current_frame.release(participant_id)
        async with self._sessions_lock:
            if self._sessions.get(participant_id) is state:
                self._sessions.pop(participant_id)
        logger.info(
            "recording finalized pid={!r} status={} session={}",
            participant_id,
            status,
            state.directory,
        )

    async def _wait_for_media(self, state: _Session) -> None:
        # CaptureTools only sends commands. Its published manifest is the
        # completion boundary; starting again during finalization drops starts.
        def read_manifest() -> tuple[Path, dict[str, Any]] | None:
            for path in state.media_directory.glob("*/manifest.json"):
                return path, json.loads(path.read_text(encoding="utf-8"))
            return None

        async def wait_for_manifest() -> tuple[Path, dict[str, Any]]:
            while (result := await asyncio.to_thread(read_manifest)) is None:
                await asyncio.sleep(0.05)
            return result

        try:
            try:
                async with asyncio.timeout(_MEDIA_FINALIZE_WARNING_S):
                    result = await wait_for_manifest()
            except TimeoutError:
                message = (
                    "shared capture is still finalizing; "
                    "keeping recording restart blocked until its manifest arrives"
                )
                logger.warning("media capture pending pid={!r}: {}", state.participant_id, message)
                await self._error(state, "media_capture_pending", message)
                result = await wait_for_manifest()
            path, manifest = result
            state.media_manifest = str(path)
            if manifest.get("complete") is not True:
                raise ValueError(f"incomplete media capture: {manifest.get('incomplete_reason')}")
            state.media_status = "complete"
        except (OSError, ValueError) as exc:
            state.media_status = "finalization_failed"
            message = str(exc)
            logger.warning("media capture finalization pid={!r}: {}", state.participant_id, message)
            await self._error(state, "media_capture_finalize", message)
