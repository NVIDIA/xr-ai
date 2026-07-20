# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed operations over live frames and recorded video chunks."""

import asyncio
import time
from pathlib import Path
from typing import Protocol

from xr_ai_nat.functions._rpc import RPCError
from xr_ai_nat.functions.video_memory.schemas import (
    EmptyRequest,
    FrameAtTimeRequest,
    QueryVideoRequest,
    VideoStatsRequest,
)

from .frames import decode_h264, live_frame_to_rgb, nv12_to_rgb, save_png
from .store import ChunkStore, safe_name


class FrameProvider(Protocol):
    def participants(self) -> list[str]: ...

    async def fetch_latest(self, participant_id: str): ...


class VideoMemoryService:
    def __init__(
        self,
        provider: FrameProvider,
        store: ChunkStore | None,
        out_dir: Path,
        gpu_id: int,
    ) -> None:
        self._provider = provider
        self._store = store
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_id = gpu_id

    async def dispatch(self, operation: str, arguments: dict) -> dict:
        if operation == "get_health":
            EmptyRequest.model_validate(arguments)
            return {"ready": True, "recording_enabled": self._store is not None}
        if operation == "list_live_participants":
            EmptyRequest.model_validate(arguments)
            return {"participants": self._provider.participants()}
        if operation == "list_recorded_participants":
            EmptyRequest.model_validate(arguments)
            participants = [] if self._store is None else await asyncio.to_thread(self._store.participants)
            return {"participants": participants}
        if operation == "get_video_stats":
            request = VideoStatsRequest.model_validate(arguments)
            store = self._require_store()
            return await asyncio.to_thread(store.stats, request.participant_id)
        if operation == "query_video":
            request = QueryVideoRequest.model_validate(arguments)
            store = self._require_store()
            data = await asyncio.to_thread(
                store.query,
                request.participant_id,
                request.start_us,
                request.end_us,
            )
            path = self._out_dir / (
                f"{safe_name(request.participant_id)}_{request.start_us}_{request.end_us}.264"
            )
            await asyncio.to_thread(path.write_bytes, data)
            return {
                "path": str(path),
                "size": len(data),
                "start_us": request.start_us,
                "end_us": request.end_us,
            }
        if operation == "get_frame_from_time":
            request = FrameAtTimeRequest.model_validate(arguments)
            if request.reference_time_us == 0 and request.second_ago == 0:
                return await self._live_frame(request)
            return await self._recorded_frame(request)
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")

    def _require_store(self) -> ChunkStore:
        if self._store is None:
            raise RPCError("recording disabled", code="recording_disabled")
        return self._store

    async def _live_frame(self, request: FrameAtTimeRequest) -> dict:
        frame = await self._provider.fetch_latest(request.participant_id)
        if frame is None:
            raise RPCError(
                f"No live frame available for {request.participant_id!r}",
                code="not_found",
            )
        try:
            rgb = await asyncio.to_thread(live_frame_to_rgb, frame)
        except ValueError as error:
            raise RPCError(str(error), code="unsupported_format") from error
        path = self._out_dir / (
            f"{safe_name(request.participant_id)}_ago0_{frame.pts_us}.png"
        )
        await asyncio.to_thread(save_png, rgb, path)
        now_us = time.time_ns() // 1_000
        return {
            "path": str(path),
            "width": frame.width,
            "height": frame.height,
            "timestamp_us": frame.pts_us,
            "second_ago": 0,
            "actual_second_ago": (now_us - frame.pts_us) / 1_000_000,
        }

    async def _recorded_frame(self, request: FrameAtTimeRequest) -> dict:
        store = self._require_store()
        now_us = time.time_ns() // 1_000
        anchor_us = request.reference_time_us or now_us
        target_us = anchor_us - request.second_ago * 1_000_000
        chunk, metadata = await asyncio.to_thread(
            store.frame_chunk,
            request.participant_id,
            target_us,
        )
        data = await asyncio.to_thread(chunk.read_bytes)
        try:
            frames = await asyncio.to_thread(decode_h264, data, self._gpu_id)
        except Exception as error:
            raise RPCError(f"Decode failed: {error}", code="decode_error") from error
        if not frames:
            raise RPCError(f"Chunk {chunk.name} decoded zero frames", code="decode_error")

        start_us = int(metadata.get("start_us", chunk.stem))
        end_us = int(metadata.get("end_us", start_us))
        declared_frames = int(metadata.get("num_frames", len(frames)))
        if declared_frames <= 1 or end_us <= start_us:
            index = 0
        else:
            ratio = (target_us - start_us) / (end_us - start_us)
            index = max(0, min(declared_frames - 1, round(ratio * (declared_frames - 1))))
        index = min(index, len(frames) - 1)
        width = int(metadata.get("width", frames[index].shape[1]))
        height = int(metadata.get("height", frames[index].shape[0] * 2 // 3))
        rgb = await asyncio.to_thread(nv12_to_rgb, frames[index], width, height)
        timestamp_us = (
            start_us
            if declared_frames <= 1
            else start_us + index * (end_us - start_us) // max(declared_frames - 1, 1)
        )
        path = self._out_dir / (
            f"{safe_name(request.participant_id)}_ago{request.second_ago}_{target_us}.png"
        )
        await asyncio.to_thread(save_png, rgb, path)
        return {
            "path": str(path),
            "width": width,
            "height": height,
            "timestamp_us": timestamp_us,
            "second_ago": request.second_ago,
            "actual_second_ago": (now_us - timestamp_us) / 1_000_000,
        }
