# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent capture process over normalized media-hub IPC."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec

from loguru import logger
from xr_ai_hub import (
    AudioChunk,
    DataMessage,
    FrameSignal,
    ParticipantEvent,
    ProcessorEndpoint,
    Subscribe,
)
from xr_ai_hub._capture import (
    CAPTURE_OBSERVATION_TOPIC,
    CAPTURE_START_TOPIC,
    CAPTURE_STOP_TOPIC,
    CAPTURE_STT_TOPIC,
    CAPTURE_TTS_TOPIC,
)

from ._recorder import SessionRecorder
from ._return_subscriber import ReturnTrafficSubscriber
from .config import CaptureConfig
from .renderer import CaptureRenderer

_RETURN_TRAFFIC_DRAIN_TIMEOUT_S = 30.0
_RETURN_TRAFFIC_DRAIN_TIMEOUT_REASON = "return_traffic_drain_timeout"
_P = ParamSpec("_P")


def _invalid_audio_reason(chunk: AudioChunk) -> str | None:
    if chunk.sample_rate <= 0:
        return "sample rate must be positive"
    if chunk.samples <= 0 or chunk.channels <= 0:
        return "sample and channel counts must be positive"
    expected_bytes = chunk.samples * chunk.channels * 4
    if len(chunk.data) != expected_bytes:
        return f"expected {expected_bytes} PCM bytes, got {len(chunk.data)}"
    return None


def _json_object(data: bytes, *, label: str, allow_empty: bool = False) -> dict[str, Any]:
    if allow_empty and not data:
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class _StartRequest:
    target: str | None
    metadata: dict[str, Any]


def _start_request(data: bytes) -> _StartRequest:
    value = _json_object(data, label="capture start command", allow_empty=True)
    unknown = set(value) - {"target", "metadata"}
    if unknown:
        raise ValueError(f"capture start command has unknown fields: {sorted(unknown)}")
    target = value.get("target")
    if target is not None and (not isinstance(target, str) or not target):
        raise ValueError("capture start target must be a non-empty string")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("capture start metadata must be a JSON object")
    return _StartRequest(target=target, metadata=metadata)


class _FrameWorker:
    """Coalesce one video track before requesting pixels and invoking NVENC."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        recorder: SessionRecorder,
        executor: ThreadPoolExecutor,
        participant_id: str,
        track_id: str,
        queue_size: int,
        sample_fps: float,
        on_failure,
    ) -> None:
        self.participant_id = participant_id
        self.track_id = track_id
        self._endpoint = endpoint
        self._recorder = recorder
        self._executor = executor
        self._queue: asyncio.Queue[FrameSignal | None] = asyncio.Queue(maxsize=queue_size)
        self._min_interval_us = round(1_000_000 / sample_fps)
        self._last_request_pts_us: int | None = None
        self._on_failure = on_failure
        self._task = asyncio.create_task(
            self._run(),
            name=f"capture-video-{participant_id}-{track_id}",
        )
        self._task.add_done_callback(on_failure)

    def submit(self, signal: FrameSignal) -> None:
        if self._queue.full():
            dropped = self._queue.get_nowait()
            if dropped is not None:
                self._recorder.note_video_drop(dropped.participant_id, dropped.pts_us)
        self._queue.put_nowait(signal)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            signal = await self._queue.get()
            if signal is None:
                return
            if (
                self._last_request_pts_us is not None
                and signal.pts_us - self._last_request_pts_us < self._min_interval_us
            ):
                continue
            self._last_request_pts_us = signal.pts_us
            frame = await self._endpoint.request_frame(signal)
            if frame is None:
                self._recorder.note_video_drop(signal.participant_id, signal.pts_us)
                continue
            await loop.run_in_executor(self._executor, self._recorder.record_video, frame)

    async def close(self) -> None:
        self._task.remove_done_callback(self._on_failure)
        if self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(None)
        await self._task


class CaptureService:
    """Coordinate hub subscriptions and the disk/NVENC capture workers."""

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._endpoint = ProcessorEndpoint(
            sub_addr=config.hub_sub_addr,
            push_addr=config.hub_push_addr,
            filter=Subscribe.ALL,
            agent_id="media-capture",
            announces_readiness=False,
        )
        self._returns = ReturnTrafficSubscriber(config.hub_sub_addr)
        self._recorder = SessionRecorder(config)
        self._renderer = (
            CaptureRenderer(
                gpu_id=config.gpu_id,
                bitrate=config.bitrate,
                overlay_seconds=config.overlay_seconds,
                overlay_lines=config.overlay_lines,
            )
            if config.profile == "demo"
            else None
        )
        self._video_executor = ThreadPoolExecutor(
            max_workers=config.encoder_workers,
            thread_name_prefix="capture-nvenc",
        )
        self._writer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="capture-writer",
        )
        self._renderer_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="capture-renderer",
            )
            if self._renderer is not None
            else None
        )
        self._frame_workers: dict[tuple[str, str], _FrameWorker] = {}
        self._departed_participants: set[str] = set()
        self._closing_participants: set[str] = set()
        self._callback_tasks: set[asyncio.Task] = set()
        self._finalize_lock = asyncio.Lock()
        self._endpoint_task: asyncio.Task | None = None
        self._return_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._failure: asyncio.Future[None] | None = None
        self._stopped = False

        self._endpoint.on_frame(self._tracked_callback(self._on_frame))
        self._endpoint.on_audio(self._tracked_callback(self._on_device_audio))
        self._endpoint.on_data(self._tracked_callback(self._on_device_data))
        self._endpoint.on_participant(self._tracked_callback(self._on_participant))
        self._returns.on_audio(self._tracked_callback(self._on_agent_audio))
        self._returns.on_data(self._tracked_callback(self._on_agent_data))
        self._returns.on_flush(self._tracked_callback(self._on_agent_flush))

    def _tracked_callback(
        self,
        callback: Callable[_P, Awaitable[None]],
    ) -> Callable[_P, Awaitable[None]]:
        async def tracked(*args: _P.args, **kwargs: _P.kwargs) -> None:
            task = asyncio.current_task()
            if task is not None:
                self._callback_tasks.add(task)
            try:
                await callback(*args, **kwargs)
            finally:
                if task is not None:
                    self._callback_tasks.discard(task)

        return tracked

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._failure = loop.create_future()
        self._endpoint_task = asyncio.create_task(self._endpoint.run(), name="capture-hub")
        self._return_task = asyncio.create_task(self._returns.run(), name="capture-return")
        for task in (self._endpoint_task, self._return_task):
            task.add_done_callback(self._task_failed)
        await self._endpoint.wait_until_running()
        # The global participant SUBSCRIBE must cross ZeroMQ's slow-joiner
        # window before readiness. A roster replay then covers anyone who
        # connected while the subscription was settling.
        await asyncio.sleep(0.1)
        await self._endpoint.request_roster()
        logger.info("media capture ready out_dir={}", self._config.out_dir)

    async def run(self) -> None:
        if self._failure is None:
            raise RuntimeError("capture service has not been started")
        await self._failure

    def _task_failed(self, task: asyncio.Task) -> None:
        if self._stopped or task.cancelled() or self._failure is None or self._failure.done():
            return
        error = task.exception()
        if error is None:
            error = RuntimeError(f"capture task {task.get_name()} stopped unexpectedly")
        self._failure.set_exception(error)

    async def _write(self, function, *args) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            self._writer_executor,
            function,
            *args,
        )

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            self._departed_participants.discard(event.participant_id)
            if self._config.session_mode == "participant":
                await self._write(
                    self._recorder.begin_session,
                    event.participant_id,
                    event.pts_us,
                )
            return
        incomplete_reason = None
        try:
            await asyncio.wait_for(
                self._returns.wait_for_departure(event),
                timeout=_RETURN_TRAFFIC_DRAIN_TIMEOUT_S,
            )
        except TimeoutError:
            incomplete_reason = _RETURN_TRAFFIC_DRAIN_TIMEOUT_REASON
            logger.warning(
                "media capture timed out draining return traffic pid={!r}; "
                "finalizing an incomplete bundle",
                event.participant_id,
            )
        self._departed_participants.add(event.participant_id)
        await self._finish_session(
            event.participant_id,
            event.pts_us,
            incomplete_reason=incomplete_reason,
        )

    async def _finish_session(
        self,
        participant_id: str,
        pts_us: int,
        *,
        incomplete_reason: str | None = None,
    ) -> None:
        if participant_id in self._closing_participants:
            return
        self._closing_participants.add(participant_id)
        try:
            workers = [
                (key, worker)
                for key, worker in self._frame_workers.items()
                if key[0] == participant_id
            ]
            for key, worker in workers:
                self._frame_workers.pop(key, None)
                await worker.close()
            async with self._finalize_lock:
                bundle = await self._write(
                    self._recorder.end_session,
                    participant_id,
                    pts_us,
                    incomplete_reason,
                )
                if bundle is not None:
                    await self._render(bundle)
                await self._write(self._recorder._prune_artifacts)
        finally:
            self._closing_participants.discard(participant_id)

    async def _start_recording(
        self,
        participant_id: str,
        pts_us: int,
        target: str | None,
        metadata: dict[str, Any],
    ) -> None:
        if self._config.session_mode != "explicit":
            logger.warning("media capture ignored explicit start in participant session mode")
            return
        self._closing_participants.discard(participant_id)
        await self._write(
            self._recorder.begin_session,
            participant_id,
            pts_us,
            "agent",
            target,
            metadata,
        )

    async def _stop_recording(self, participant_id: str, pts_us: int) -> None:
        if self._config.session_mode != "explicit":
            logger.warning("media capture ignored explicit stop in participant session mode")
            return
        await self._finish_session(participant_id, pts_us)

    def _render_bundle(self, bundle: Path) -> None:
        if self._renderer is None:
            return
        try:
            self._renderer.render(bundle)
        except Exception as exc:
            logger.warning("media capture rendering failed path={}: {}", bundle, exc)

    async def _render(self, bundle: Path) -> None:
        if self._renderer_executor is None:
            return
        future = asyncio.get_running_loop().run_in_executor(
            self._renderer_executor,
            self._render_bundle,
            bundle,
        )
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            _ = await future
            raise

    async def _drain_callbacks(self) -> None:
        current = asyncio.current_task()
        while pending := {
            task
            for task in self._callback_tasks
            if task is not current and not task.done()
        }:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _on_frame(self, signal: FrameSignal) -> None:
        if (
            signal.participant_id in self._departed_participants
            or signal.participant_id in self._closing_participants
            or not self._recorder.has_session(signal.participant_id)
        ):
            return
        key = (signal.participant_id, signal.track_id)
        worker = self._frame_workers.get(key)
        if worker is None:
            worker = _FrameWorker(
                endpoint=self._endpoint,
                recorder=self._recorder,
                executor=self._video_executor,
                participant_id=signal.participant_id,
                track_id=signal.track_id,
                queue_size=self._config.frame_queue_size,
                sample_fps=self._config.sample_fps,
                on_failure=self._task_failed,
            )
            self._frame_workers[key] = worker
        worker.submit(signal)

    async def _on_device_audio(self, chunk: AudioChunk) -> None:
        await self._record_audio("device", chunk)

    async def _on_agent_audio(self, chunk: AudioChunk) -> None:
        await self._record_audio("agent", chunk)

    async def _record_audio(self, direction: str, chunk: AudioChunk) -> None:
        if (
            chunk.participant_id in self._departed_participants
            or chunk.participant_id in self._closing_participants
            or not self._recorder.has_session(chunk.participant_id)
        ):
            return
        reason = _invalid_audio_reason(chunk)
        if reason is not None:
            logger.warning(
                "media capture dropped invalid {} audio pid={!r}: {}",
                direction,
                chunk.participant_id,
                reason,
            )
            return
        await self._write(self._recorder.record_audio, direction, chunk)

    async def _on_device_data(self, message: DataMessage) -> None:
        if (
            message.participant_id in self._departed_participants
            or message.participant_id in self._closing_participants
            or not self._recorder.has_session(message.participant_id)
        ):
            return
        await self._write(self._recorder.record_data, "device", message)

    async def _on_agent_data(self, message: DataMessage) -> None:
        if (
            message.participant_id in self._departed_participants
            or message.participant_id in self._closing_participants
        ):
            return
        if message.topic == CAPTURE_START_TOPIC:
            try:
                request = _start_request(message.data)
                await self._start_recording(
                    message.participant_id,
                    message.pts_us,
                    request.target,
                    request.metadata,
                )
            except ValueError as exc:
                logger.warning("media capture ignored invalid start command: {}", exc)
            return
        if message.topic == CAPTURE_STOP_TOPIC:
            await self._stop_recording(message.participant_id, message.pts_us)
            return
        if not self._recorder.has_session(message.participant_id):
            return
        if message.topic == CAPTURE_OBSERVATION_TOPIC:
            try:
                observation = _json_object(message.data, label="capture observation")
            except ValueError as exc:
                logger.warning("media capture ignored invalid observation: {}", exc)
                return
            await self._write(self._recorder.record_observation, message, observation)
            return
        if message.topic == CAPTURE_STT_TOPIC:
            await self._write(self._recorder.record_voice_caption, "user", message)
            return
        if message.topic == CAPTURE_TTS_TOPIC:
            await self._write(self._recorder.record_voice_caption, "agent", message)
            return
        await self._write(self._recorder.record_data, "agent", message)

    async def _on_agent_flush(self, flush) -> None:
        if (
            flush.participant_id in self._departed_participants
            or flush.participant_id in self._closing_participants
            or not self._recorder.has_session(flush.participant_id)
        ):
            return
        await self._write(
            self._recorder.record_flush,
            flush.participant_id,
            time.time_ns() // 1_000,
        )

    async def stop(self) -> None:
        if self._stop_task is None:
            self._stopped = True
            self._stop_task = asyncio.create_task(
                self._stop_impl(),
                name="capture-shutdown",
            )
        try:
            await asyncio.shield(self._stop_task)
        except asyncio.CancelledError:
            await self._stop_task
            raise

    async def _stop_impl(self) -> None:
        self._endpoint.stop()
        self._returns.stop()
        for task in (self._endpoint_task,):
            if task is not None and not task.done():
                task.cancel()
        if (
            self._return_task is not None
            and not self._return_task.done()
            and self._return_task not in self._callback_tasks
        ):
            self._return_task.cancel()
        await asyncio.gather(
            *(task for task in (self._endpoint_task, self._return_task) if task is not None),
            return_exceptions=True,
        )
        await asyncio.sleep(0)
        await self._drain_callbacks()
        await asyncio.gather(
            *(worker.close() for worker in self._frame_workers.values()),
            return_exceptions=True,
        )
        self._frame_workers.clear()
        async with self._finalize_lock:
            completed = await self._write(self._recorder.close)
            for bundle in completed:
                await self._render(bundle)
            await self._write(self._recorder._prune_artifacts)
        self._writer_executor.shutdown(wait=True, cancel_futures=False)
        if self._renderer_executor is not None:
            self._renderer_executor.shutdown(wait=True, cancel_futures=False)
        self._video_executor.shutdown(wait=True, cancel_futures=False)
        self._endpoint.close()
        self._returns.close()
