# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent capture process over normalized media-hub IPC."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from xr_ai_hub import (
    AudioChunk,
    DataMessage,
    FrameSignal,
    ParticipantEvent,
    ProcessorEndpoint,
    Subscribe,
)
from xr_ai_hub._capture import CAPTURE_STT_TOPIC, CAPTURE_TTS_TOPIC

from ._recorder import SessionRecorder
from ._return_subscriber import ReturnTrafficSubscriber
from .config import CaptureConfig


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
        on_failure,
    ) -> None:
        self.participant_id = participant_id
        self.track_id = track_id
        self._endpoint = endpoint
        self._recorder = recorder
        self._executor = executor
        self._queue: asyncio.Queue[FrameSignal | None] = asyncio.Queue(maxsize=queue_size)
        self._on_failure = on_failure
        self._task = asyncio.create_task(
            self._run(),
            name=f"capture-video-{participant_id}-{track_id}",
        )
        self._task.add_done_callback(on_failure)

    def submit(self, signal: FrameSignal) -> None:
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                if dropped is not None:
                    self._recorder.note_video_drop(dropped.participant_id, dropped.pts_us)
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(signal)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            signal = await self._queue.get()
            if signal is None:
                return
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
        self._video_executor = ThreadPoolExecutor(
            max_workers=config.encoder_workers,
            thread_name_prefix="capture-nvenc",
        )
        self._writer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="capture-writer",
        )
        self._frame_workers: dict[tuple[str, str], _FrameWorker] = {}
        self._departed_participants: set[str] = set()
        self._endpoint_task: asyncio.Task | None = None
        self._return_task: asyncio.Task | None = None
        self._failure: asyncio.Future[None] | None = None
        self._stopped = False

        self._endpoint.on_frame(self._on_frame)
        self._endpoint.on_audio(self._on_device_audio)
        self._endpoint.on_data(self._on_device_data)
        self._endpoint.on_participant(self._on_participant)
        self._returns.on_audio(self._on_agent_audio)
        self._returns.on_data(self._on_agent_data)
        self._returns.on_flush(self._on_agent_flush)

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

    async def _write(self, function, *args) -> None:
        await asyncio.get_running_loop().run_in_executor(
            self._writer_executor,
            function,
            *args,
        )

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            self._departed_participants.discard(event.participant_id)
            await self._write(
                self._recorder.begin_session,
                event.participant_id,
                event.pts_us,
            )
            return
        self._departed_participants.add(event.participant_id)
        workers = [
            (key, worker)
            for key, worker in self._frame_workers.items()
            if key[0] == event.participant_id
        ]
        for key, worker in workers:
            self._frame_workers.pop(key, None)
            await worker.close()
        await self._write(
            self._recorder.end_session,
            event.participant_id,
            event.pts_us,
        )

    async def _on_frame(self, signal: FrameSignal) -> None:
        if signal.participant_id in self._departed_participants:
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
                on_failure=self._task_failed,
            )
            self._frame_workers[key] = worker
        worker.submit(signal)

    async def _on_device_audio(self, chunk: AudioChunk) -> None:
        if chunk.participant_id in self._departed_participants:
            return
        await self._write(self._recorder.record_audio, "device", chunk)

    async def _on_agent_audio(self, chunk: AudioChunk) -> None:
        if chunk.participant_id in self._departed_participants:
            return
        await self._write(self._recorder.record_audio, "agent", chunk)

    async def _on_device_data(self, message: DataMessage) -> None:
        if message.participant_id in self._departed_participants:
            return
        await self._write(self._recorder.record_data, "device", message)

    async def _on_agent_data(self, message: DataMessage) -> None:
        if message.participant_id in self._departed_participants:
            return
        if message.topic == CAPTURE_STT_TOPIC:
            await self._write(self._recorder.record_voice_caption, "user", message)
            return
        if message.topic == CAPTURE_TTS_TOPIC:
            await self._write(self._recorder.record_voice_caption, "agent", message)
            return
        await self._write(self._recorder.record_data, "agent", message)

    async def _on_agent_flush(self, flush) -> None:
        if flush.participant_id in self._departed_participants:
            return
        await self._write(
            self._recorder.record_flush,
            flush.participant_id,
            time.time_ns() // 1_000,
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._endpoint.stop()
        for task in (self._endpoint_task, self._return_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._endpoint_task, self._return_task) if task is not None),
            return_exceptions=True,
        )
        self._endpoint.close()
        self._returns.close()
        await asyncio.gather(
            *(worker.close() for worker in self._frame_workers.values()),
            return_exceptions=True,
        )
        self._frame_workers.clear()
        await self._write(self._recorder.close)
        self._writer_executor.shutdown(wait=True, cancel_futures=False)
        self._video_executor.shutdown(wait=True, cancel_futures=False)
