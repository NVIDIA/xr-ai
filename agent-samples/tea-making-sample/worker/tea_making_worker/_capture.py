# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record replayable SOP runs without taking ownership of execution.

The application activates one run bundle; routing, foreground and observation
attempts attach request/step identity through context variables. Model wrappers
record the actual requests and responses, while image blobs are content-addressed
and configuration snapshots preserve the effective YAML for later analysis.

File writes are serialized and drained across cancellation. Capture must finish
its writes before an attempt or run is marked closed; it does not schedule turns,
choose routes, or decide whether workflow evidence is valid.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import nemo_relay
import yaml
from pydantic import BaseModel

_ACTIVE: ContextVar[_SOPCapture | None] = ContextVar("sop_capture", default=None)
_ATTEMPT: ContextVar[_Attempt | None] = ContextVar("sop_attempt", default=None)
_ROUTING_ATTEMPT: ContextVar[Any] = ContextVar("sop_routing_attempt", default=None)


def _json(value: Any) -> Any:
    """Convert supported typed values to JSON-safe data and redact credential fields."""
    if isinstance(value, BaseModel):
        return _json(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if str(key).lower()
            in {
                "api_key",
                "hf_token",
                "access_token",
                "refresh_token",
                "password",
                "secret",
                "headers",
            }
            else _json(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _segment(value: str) -> str:
    """Create a readable, collision-resistant directory component from an identity."""
    readable = re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip(".-")[:64] or "unnamed"
    return readable + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON snapshot after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append(path: Path, value: dict[str, Any]) -> None:
    """Append one already-serialized event to its ordered JSONL stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False) + "\n")


class _SOPCapture:
    """Own one run bundle and serialize its image, snapshot and event writes."""

    def __init__(self, output_dir: Path, *, enabled: bool = True) -> None:
        """Allocate a unique run location; actual files are created by start or recording."""
        self.enabled = enabled
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
        self.directory = output_dir / "runs" / self.run_id
        self._lock = asyncio.Lock()
        self._manifest: dict[str, Any] = {}
        self._closed = False

    async def io(self, function, *args):
        # A cancelled caller must not leave a file write racing image/session cleanup.
        """Run one serialized file operation and drain it before propagating cancellation.

        Cancelling an await cannot stop a thread already writing. Keep the lock and shield
        that thread through repeated cancellation so cleanup never races an unfinished
        write. Re-raise cancellation only after retrieving the operation's result.
        """
        async with self._lock:
            write_task = asyncio.create_task(asyncio.to_thread(function, *args))
            try:
                return await asyncio.shield(write_task)
            except asyncio.CancelledError:
                while not write_task.done():
                    try:
                        await asyncio.shield(write_task)
                    except asyncio.CancelledError:
                        continue
                write_task.result()
                raise

    async def start(self, config, workflow, *, source_revision: str | None = None) -> None:
        """Create the manifest and snapshot effective configuration plus source hashes.

        The effective worker snapshot includes all inline prompts. Workflow YAML and its
        resolved form are both retained so replay can distinguish source policy from its
        validated representation.
        """
        self._manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_at_us": time.time_ns() // 1000,
            "status": "running",
            "capture_enabled": self.enabled,
            "source_revision": source_revision,
            "workflow": {"name": workflow.name, "steps": list(workflow.steps)},
        }
        await self.io(_write_json, self.directory / "manifest.json", self._manifest)
        snapshots = {
            "worker": _json(config),
            "workflow": yaml.safe_load(config.workflow_config.read_text()),
            "resolved_workflow": _json(workflow),
            "models": json.loads(config.models_config.read_text()),
            "voice_gate": yaml.safe_load(config.voice_gate_yaml.read_text()),
        }
        configuration_hashes = {}
        for name, value in snapshots.items():
            safe_snapshot = _json(value)
            configuration_hashes[name] = hashlib.sha256(json.dumps(safe_snapshot, sort_keys=True).encode()).hexdigest()
            await self.io(_write_json, self.directory / "config" / f"{name}.json", safe_snapshot)
        self._manifest["source_sha256"] = {
            path.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.rglob("*.py")
        }
        self._manifest["config_sha256"] = configuration_hashes
        await self.io(_write_json, self.directory / "manifest.json", self._manifest)

    @asynccontextmanager
    async def activate(self):
        """Scope recording to this run and persist its final completion/failure status."""
        token = _ACTIVE.set(self)
        status = "completed"
        try:
            yield
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception:  # noqa: BLE001 -- finalize the run as failed and preserve the original failure.
            status = "failed"
            raise
        finally:
            _ACTIVE.reset(token)
            self._closed = True
            self._manifest.update(status=status, ended_at_us=time.time_ns() // 1000)
            await self.io(_write_json, self.directory / "manifest.json", self._manifest)

    async def encoded_image(self, image) -> bytes:
        """Read local encoded image data without requiring its original live source later."""
        if isinstance(image, bytes):
            return image
        if isinstance(image, Path):
            return await self.io(image.read_bytes)
        if isinstance(image, str) and image.startswith("data:"):
            return base64.b64decode(image.partition(",")[2], validate=True)
        raise ValueError("SOP capture requires local encoded image bytes, paths or data URLs")

    async def image(self, image) -> dict[str, Any]:
        """Store one content-addressed image and return its stable bundle-relative reference."""
        encoded_image = await self.encoded_image(image)
        image_digest = hashlib.sha256(encoded_image).hexdigest()
        extension = (
            "jpg" if encoded_image.startswith(b"\xff\xd8")
            else "png" if encoded_image.startswith(b"\x89PNG")
            else "bin"
        )
        relative_image_path = Path("images") / f"{image_digest}.{extension}"

        def save():
            path = self.directory / relative_image_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(encoded_image)
                temporary.replace(path)

        await self.io(save)
        return {"path": relative_image_path.as_posix(), "sha256": image_digest, "bytes": len(encoded_image)}


class _Attempt:
    """Correlate one request or observation with its participant, step and ordered events."""

    def __init__(self, capture, participant, channel, step_id, before, request_id):
        """Allocate attempt identity and location without writing an event yet."""
        self.capture = capture
        self.attempt_id = uuid4().hex
        self.frame: dict[str, Any] | None = None
        self.finished = False
        self.context = {
            "attempt_id": self.attempt_id,
            "participant_id": participant,
            "channel": channel,
            "step_id": step_id,
            "request_id": request_id,
        }
        location = Path("participants") / _segment(participant)
        location /= Path("steps") / _segment(step_id) if step_id else Path("applications")
        self.path = location / channel / self.attempt_id / "events.jsonl"

    async def event(self, kind, **data):
        """Append an event only while both its attempt and run remain open."""
        if self.capture._closed or self.finished:
            raise RuntimeError("SOP recording is closed")
        await self.capture.io(
            _append,
            self.capture.directory / self.path,
            {"schema_version": 1, "timestamp_us": time.time_ns() // 1000, **self.context, "kind": kind, **_json(data)},
        )


@asynccontextmanager
async def _attempt(
    participant: str,
    channel: str,
    *,
    step_id: str | None = None,
    snapshot: Callable[[], dict] | None = None,
    request_id: str | None = None,
):
    """Record attempt boundaries and before/after native snapshots.

    Args:
        participant: Participant owning this observation or request.
        channel: Kind of attempt, such as routing, foreground or observer.
        step_id: Active workflow step when this attempt belongs to one.
        snapshot: Optional synchronous native-state reader used at both boundaries.
        request_id: SDK request identity shared by routing and its foreground answer.

    Yields:
        Control to the caller with this attempt active for model/frame/event recording.
        When capture is disabled, delegate without opening an attempt.
    """
    capture = _ACTIVE.get()
    if capture is None or not capture.enabled:
        yield
        return
    if capture._closed:
        raise RuntimeError("SOP recording is closed")
    before = snapshot() if snapshot else {}
    attempt = _Attempt(capture, participant, channel, step_id, before, request_id)
    token = _ATTEMPT.set(attempt)
    status = "completed"
    try:
        await attempt.event("attempt.started", before=before)
        await capture.io(
            _append, capture.directory / "attempts.jsonl", {**attempt.context, "path": attempt.path.as_posix()}
        )
        try:
            yield
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status = "failed"
            await attempt.event("attempt.error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            try:
                await attempt.event("attempt.finished", status=status, after=snapshot() if snapshot else {})
            finally:
                attempt.finished = True
    finally:
        _ATTEMPT.reset(token)


async def _record_routing(status: str, participant: str, *, snapshot: Callable[[str], dict]) -> None:
    """Record SDK turn decisions using its public status hook and Relay metadata.

    The processing/idle calls occur inside the same SDK-owned turn context. Keep only
    a recording scope, never a second task lifecycle. The SDK's final idle call closes
    that scope with its original cancellation/failure information.
    """
    if status == "processing":
        capture = _ACTIVE.get()
        if capture is None or not capture.enabled:
            return
        metadata = nemo_relay.scope.get_handle().metadata or {}
        recording_scope = _attempt(participant, "routing", request_id=metadata["message_id"],
                         snapshot=lambda: snapshot(participant))
        await recording_scope.__aenter__()
        _ROUTING_ATTEMPT.set(recording_scope)
    elif status == "idle":
        recording_scope = _ROUTING_ATTEMPT.get()
        if recording_scope is not None:
            _ROUTING_ATTEMPT.set(None)
            # The SDK calls idle in its turn's finally block. Preserve failure
            # and cancellation while closing recording; no task is owned here.
            await recording_scope.__aexit__(*sys.exc_info())


async def _event(kind: str, **data) -> None:
    """Record an application event when execution currently owns an active attempt."""
    current_attempt = _ATTEMPT.get()
    if current_attempt is not None:
        await current_attempt.event(kind, **data)


async def _frame(frame) -> None:
    """Attach exact frame metadata to subsequent recorded visual calls."""
    current_attempt = _ATTEMPT.get()
    if current_attempt is not None:
        current_attempt.frame = frame.model_dump(mode="json", exclude={"image"})
        await current_attempt.event("frame.acquired", frame=current_attempt.frame, image_reference=frame.image.uri)


class _RecordedVision:
    """Record visual requests/images and delegate inference to the existing service."""

    def __init__(self, service):
        """Wrap the existing vision service without changing its invocation contract."""
        self._service = service

    async def ask_images(self, images, question, **kwargs):
        """Record one visual call with images, outcome and latency, preserving failures."""
        current_attempt = _ATTEMPT.get()
        if current_attempt is None:
            return await self._service.ask_images(images, question, **kwargs)
        images = [await current_attempt.capture.encoded_image(image) for image in images]
        image_references = [await current_attempt.capture.image(image) for image in images]
        call_id = uuid4().hex
        await current_attempt.event(
            "vision.started",
            call_id=call_id,
            images=image_references,
            frame=current_attempt.frame,
            question=question,
            settings={k: v for k, v in kwargs.items() if k != "headers"},
        )
        started_at = time.perf_counter()
        try:
            response = await self._service.ask_images(images, question, **kwargs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- record the model boundary failure and re-raise unchanged.
            await current_attempt.event(
                "vision.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc)
            )
            raise
        await current_attempt.event(
            "vision.finished",
            call_id=call_id,
            response=response,
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
        return response

    async def stream_images(self, images, question, **kwargs):
        """Yield original model chunks while retaining a complete or interrupted trace.

        The model stream is closed before the final event is written. A partial response
        is recorded with complete=false rather than presented as a completed inference.
        """
        current_attempt = _ATTEMPT.get()
        if current_attempt is None:
            async for text in self._service.stream_images(images, question, **kwargs):
                yield text
            return
        images = [await current_attempt.capture.encoded_image(image) for image in images]
        image_references = [await current_attempt.capture.image(image) for image in images]
        call_id = uuid4().hex
        await current_attempt.event(
            "vision.started",
            call_id=call_id,
            streaming=True,
            images=image_references,
            frame=current_attempt.frame,
            question=question,
            settings={k: v for k, v in kwargs.items() if k != "headers"},
        )
        response_fragments = []
        started_at = time.perf_counter()
        stream_completed = False
        stream = self._service.stream_images(images, question, **kwargs)
        try:
            async for text in stream:
                response_fragments.append(text)
                yield text
            stream_completed = True
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- retain the streaming failure and re-raise unchanged.
            await current_attempt.event(
                "vision.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc)
            )
            raise
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
            await current_attempt.event(
                "vision.finished",
                call_id=call_id,
                content="".join(response_fragments),
                complete=stream_completed,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )


class _RecordedDecisions:
    """Record bounded decision requests and scores without modifying their policy."""

    def __init__(self, service):
        """Wrap the existing decision service without changing its invocation contract."""
        self._service = service

    async def decide(self, state, questions, **kwargs):
        """Record decision inputs, optional images, result and elapsed inference time."""
        current_attempt = _ATTEMPT.get()
        if current_attempt is None:
            return await self._service.decide(state, questions, **kwargs)
        call_id = uuid4().hex
        await current_attempt.event(
            "decision.started",
            call_id=call_id,
            state=state,
            questions=questions,
            images=[await current_attempt.capture.image(image) for image in kwargs.get("images", ())],
            settings={k: v for k, v in kwargs.items() if k != "images"},
        )
        started_at = time.perf_counter()
        try:
            response = await self._service.decide(state, questions, **kwargs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- record the model boundary failure and re-raise unchanged.
            await current_attempt.event(
                "decision.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc)
            )
            raise
        await current_attempt.event(
            "decision.finished",
            call_id=call_id,
            response=response,
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
        return response


class _RecordedLanguage:
    """Record real language requests, including SDK context-compaction calls."""

    def __init__(self, service):
        """Wrap the existing language service without changing its invocation contract."""
        self._service = service

    async def chat(self, messages, **kwargs):
        """Record the actual language request and response, preserving failure/cancellation."""
        current_attempt = _ATTEMPT.get()
        if current_attempt is None:
            return await self._service.chat(messages, **kwargs)
        call_id = uuid4().hex
        await current_attempt.event(
            "language.started",
            call_id=call_id,
            messages=messages,
            settings={k: v for k, v in kwargs.items() if k != "headers"},
        )
        started_at = time.perf_counter()
        try:
            response = await self._service.chat(messages, **kwargs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- record the model boundary failure and re-raise unchanged.
            await current_attempt.event(
                "language.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc)
            )
            raise
        await current_attempt.event(
            "language.finished",
            call_id=call_id,
            response=response,
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
        return response
