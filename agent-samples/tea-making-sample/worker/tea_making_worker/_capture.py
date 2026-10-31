# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned SOP run bundles, attempts, and exact visual/decision evidence."""

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
    readable = re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip(".-")[:64] or "unnamed"
    return readable + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False) + "\n")


class _SOPCapture:
    def __init__(self, output_dir: Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
        self.directory = output_dir / "runs" / self.run_id
        self._lock = asyncio.Lock()
        self._manifest: dict[str, Any] = {}
        self._closed = False

    async def io(self, function, *args):
        # A cancelled caller must not leave a file write racing image/session cleanup.
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(function, *args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                task.result()
                raise

    async def start(self, config, workflow, *, source_revision: str | None = None) -> None:
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
        hashes = {}
        for name, value in snapshots.items():
            safe = _json(value)
            hashes[name] = hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest()
            await self.io(_write_json, self.directory / "config" / f"{name}.json", safe)
        prompts = Path(__file__).parent / "prompts"
        for path in prompts.iterdir():
            if path.is_file():
                await self.io(
                    _write_json, self.directory / "config" / "prompts" / f"{path.name}.json", {"text": path.read_text()}
                )
        self._manifest["source_sha256"] = {
            path.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.rglob("*.py")
        }
        self._manifest["config_sha256"] = hashes
        await self.io(_write_json, self.directory / "manifest.json", self._manifest)

    @asynccontextmanager
    async def activate(self):
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
        if isinstance(image, bytes):
            return image
        if isinstance(image, Path):
            return await self.io(image.read_bytes)
        if isinstance(image, str) and image.startswith("data:"):
            return base64.b64decode(image.partition(",")[2], validate=True)
        raise ValueError("SOP capture requires local encoded image bytes, paths or data URLs")

    async def image(self, image) -> dict[str, Any]:
        data = await self.encoded_image(image)
        digest = hashlib.sha256(data).hexdigest()
        extension = "jpg" if data.startswith(b"\xff\xd8") else "png" if data.startswith(b"\x89PNG") else "bin"
        relative = Path("images") / f"{digest}.{extension}"

        def save():
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)

        await self.io(save)
        return {"path": relative.as_posix(), "sha256": digest, "bytes": len(data)}


class _Attempt:
    def __init__(self, capture, participant, channel, step_id, before, request_id):
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
    """Record decisions inside the SDK-owned turn, using its public Relay metadata."""
    if status == "processing":
        capture = _ACTIVE.get()
        if capture is None or not capture.enabled:
            return
        metadata = nemo_relay.scope.get_handle().metadata or {}
        scope = _attempt(participant, "routing", request_id=metadata["message_id"],
                         snapshot=lambda: snapshot(participant))
        await scope.__aenter__()
        _ROUTING_ATTEMPT.set(scope)
    elif status == "idle":
        scope = _ROUTING_ATTEMPT.get()
        if scope is not None:
            _ROUTING_ATTEMPT.set(None)
            # The SDK calls idle in its turn's finally block. Preserve failure
            # and cancellation while closing recording; no task is owned here.
            await scope.__aexit__(*sys.exc_info())


async def _event(kind: str, **data) -> None:
    current = _ATTEMPT.get()
    if current is not None:
        await current.event(kind, **data)


async def _frame(frame) -> None:
    current = _ATTEMPT.get()
    if current is not None:
        current.frame = frame.model_dump(mode="json", exclude={"image"})
        await current.event("frame.acquired", frame=current.frame, image_reference=frame.image.uri)


class _RecordedVision:
    def __init__(self, service):
        self._service = service

    async def ask_images(self, images, question, **kwargs):
        current = _ATTEMPT.get()
        if current is None:
            return await self._service.ask_images(images, question, **kwargs)
        images = [await current.capture.encoded_image(image) for image in images]
        references = [await current.capture.image(image) for image in images]
        call_id = uuid4().hex
        await current.event(
            "vision.started",
            call_id=call_id,
            images=references,
            frame=current.frame,
            question=question,
            settings={k: v for k, v in kwargs.items() if k != "headers"},
        )
        started = time.perf_counter()
        try:
            response = await self._service.ask_images(images, question, **kwargs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- record the model boundary failure and re-raise unchanged.
            await current.event("vision.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc))
            raise
        await current.event(
            "vision.finished", call_id=call_id, response=response, duration_ms=(time.perf_counter() - started) * 1000
        )
        return response

    async def stream_images(self, images, question, **kwargs):
        current = _ATTEMPT.get()
        if current is None:
            async for text in self._service.stream_images(images, question, **kwargs):
                yield text
            return
        images = [await current.capture.encoded_image(image) for image in images]
        references = [await current.capture.image(image) for image in images]
        call_id = uuid4().hex
        await current.event(
            "vision.started",
            call_id=call_id,
            streaming=True,
            images=references,
            frame=current.frame,
            question=question,
            settings={k: v for k, v in kwargs.items() if k != "headers"},
        )
        fragments = []
        started = time.perf_counter()
        complete = False
        stream = self._service.stream_images(images, question, **kwargs)
        try:
            async for text in stream:
                fragments.append(text)
                yield text
            complete = True
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- retain the streaming failure and re-raise unchanged.
            await current.event("vision.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
            await current.event(
                "vision.finished",
                call_id=call_id,
                content="".join(fragments),
                complete=complete,
                duration_ms=(time.perf_counter() - started) * 1000,
            )


class _RecordedDecisions:
    def __init__(self, service):
        self._service = service

    async def decide(self, state, questions, **kwargs):
        current = _ATTEMPT.get()
        if current is None:
            return await self._service.decide(state, questions, **kwargs)
        call_id = uuid4().hex
        await current.event(
            "decision.started",
            call_id=call_id,
            state=state,
            questions=questions,
            images=[await current.capture.image(image) for image in kwargs.get("images", ())],
            settings={k: v for k, v in kwargs.items() if k != "images"},
        )
        started = time.perf_counter()
        try:
            response = await self._service.decide(state, questions, **kwargs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- record the model boundary failure and re-raise unchanged.
            await current.event("decision.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc))
            raise
        await current.event(
            "decision.finished", call_id=call_id, response=response, duration_ms=(time.perf_counter() - started) * 1000
        )
        return response


class _RecordedLanguage:
    def __init__(self, service):
        self._service = service

    async def chat(self, messages, **kwargs):
        current = _ATTEMPT.get()
        if current is None:
            return await self._service.chat(messages, **kwargs)
        call_id = uuid4().hex
        await current.event(
            "language.started",
            call_id=call_id,
            messages=messages,
            settings={k: v for k, v in kwargs.items() if k != "headers"},
        )
        started = time.perf_counter()
        try:
            response = await self._service.chat(messages, **kwargs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- record the model boundary failure and re-raise unchanged.
            await current.event("language.failed", call_id=call_id, error_type=type(exc).__name__, error=str(exc))
            raise
        await current.event(
            "language.finished", call_id=call_id, response=response, duration_ms=(time.perf_counter() - started) * 1000
        )
        return response
