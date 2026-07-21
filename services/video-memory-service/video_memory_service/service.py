# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed operations over recorded video chunks."""

import asyncio
from pathlib import Path

from xr_ai_nat.functions._rpc import RPCError
from xr_ai_nat.functions.video_memory.schemas import (
    EmptyRequest,
    HistoricalFrameRequest,
    QueryVideoRequest,
    VideoStatsRequest,
)

from .frames import decode_h264, nv12_to_rgb, save_png
from .store import ChunkStore, safe_name


def select_decoded_frame(
    *,
    start_us: int,
    end_us: int,
    declared_frames: int,
    decoded_frames: int,
    target_us: int,
) -> tuple[int, int]:
    """Choose the nearest available decoded frame and its metadata timestamp."""
    if decoded_frames <= 0:
        raise ValueError("decoded_frames must be positive")
    if declared_frames <= 1 or end_us <= start_us:
        return 0, start_us

    ratio = (target_us - start_us) / (end_us - start_us)
    declared_index = max(
        0,
        min(declared_frames - 1, round(ratio * (declared_frames - 1))),
    )
    index = min(declared_index, decoded_frames - 1)
    timestamp_us = start_us + index * (end_us - start_us) // (declared_frames - 1)
    return index, timestamp_us


class VideoMemoryService:
    def __init__(
        self,
        store: ChunkStore | None,
        out_dir: Path,
        gpu_id: int,
    ) -> None:
        self._store = store
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_id = gpu_id

    async def dispatch(self, operation: str, arguments: dict) -> dict:
        if operation == "get_health":
            EmptyRequest.model_validate(arguments)
            return {"ready": True, "recording_enabled": self._store is not None}
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
            request = HistoricalFrameRequest.model_validate(arguments)
            return await self._recorded_frame(request)
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")

    def _require_store(self) -> ChunkStore:
        if self._store is None:
            raise RPCError("recording disabled", code="recording_disabled")
        return self._store

    async def _recorded_frame(self, request: HistoricalFrameRequest) -> dict:
        store = self._require_store()
        target_us = request.reference_time_us - request.second_ago * 1_000_000
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
        index, timestamp_us = select_decoded_frame(
            start_us=start_us,
            end_us=end_us,
            declared_frames=declared_frames,
            decoded_frames=len(frames),
            target_us=target_us,
        )
        width = int(metadata.get("width", frames[index].shape[1]))
        height = int(metadata.get("height", frames[index].shape[0] * 2 // 3))
        rgb = await asyncio.to_thread(nv12_to_rgb, frames[index], width, height)
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
            "actual_second_ago": (request.reference_time_us - timestamp_us) / 1_000_000,
        }
