# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed operations over recorded video chunks."""

import asyncio
import logging
from pathlib import Path

from pydantic import ValidationError
from xr_ai_tools.rpc import RPCError
from xr_ai_tools.types import EmptyRequest
from xr_ai_tools.video_memory import (
    HistoricalFrameRequest,
    HistoricalFramesRequest,
    HistoricalVideoRequest,
    LatestFramesRequest,
    LatestVideoRequest,
    VideoStatsRequest,
)

from .frames import decode_h264, nv12_to_rgb, save_png
from .store import ChunkStore, safe_name

_LOGGER = logging.getLogger(__name__)


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


def sample_target_timestamps(start_us: int, end_us: int, frame_budget: int) -> list[int]:
    """Return ordered targets spanning the window within one total budget."""
    if frame_budget <= 0:
        raise ValueError("frame_budget must be positive")
    if frame_budget == 1:
        return [end_us]
    return [
        start_us + index * (end_us - start_us) // (frame_budget - 1)
        for index in range(frame_budget)
    ]


def _chunk_bounds(path: Path, metadata: dict) -> tuple[int, int]:
    start_us = int(metadata.get("start_us", path.stem))
    return start_us, int(metadata.get("end_us", start_us))


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
        try:
            return await self._dispatch(operation, arguments)
        except ValidationError as exc:
            raise RPCError(str(exc), code="invalid_request") from exc

    async def _dispatch(self, operation: str, arguments: dict) -> dict:
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
        if operation == "get_latest_video":
            request = LatestVideoRequest.model_validate(arguments)
            start_us, end_us = await self._latest_window(
                request.participant_id,
                request.duration_seconds,
            )
            return await self._recorded_video(request.participant_id, start_us, end_us)
        if operation == "get_latest_frames":
            request = LatestFramesRequest.model_validate(arguments)
            start_us, end_us = await self._latest_window(
                request.participant_id,
                request.duration_seconds,
            )
            return await self._sample_frames(request, start_us, end_us)
        if operation == "get_historical_frame":
            request = HistoricalFrameRequest.model_validate(arguments)
            return await self._recorded_frame(request)
        if operation == "get_historical_video":
            request = HistoricalVideoRequest.model_validate(arguments)
            start_us, end_us = self._historical_window(
                request.start_us,
                request.duration_seconds,
            )
            return await self._recorded_video(request.participant_id, start_us, end_us)
        if operation == "get_historical_frames":
            request = HistoricalFramesRequest.model_validate(arguments)
            start_us, end_us = self._historical_window(
                request.start_us,
                request.duration_seconds,
            )
            return await self._sample_frames(request, start_us, end_us)
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")

    def _require_store(self) -> ChunkStore:
        if self._store is None:
            raise RPCError("recording disabled", code="recording_disabled")
        return self._store

    async def _latest_window(
        self,
        participant_id: str,
        duration_seconds: int,
    ) -> tuple[int, int]:
        stats = await asyncio.to_thread(self._require_store().stats, participant_id)
        end_us = int(stats["latest_us"])
        start_us = max(
            int(stats["earliest_us"]),
            end_us - duration_seconds * 1_000_000,
        )
        return start_us, end_us

    @staticmethod
    def _historical_window(start_us: int, duration_seconds: int) -> tuple[int, int]:
        return start_us, start_us + duration_seconds * 1_000_000

    async def _recorded_video(
        self,
        participant_id: str,
        start_us: int,
        end_us: int,
    ) -> dict:
        data = await asyncio.to_thread(
            self._require_store().query,
            participant_id,
            start_us,
            end_us,
        )
        path = self._out_dir / f"{safe_name(participant_id)}_{start_us}_{end_us}.264"
        await asyncio.to_thread(path.write_bytes, data)
        return {
            "path": str(path),
            "size": len(data),
            "start_us": start_us,
            "end_us": end_us,
        }

    async def _recorded_frame(self, request: HistoricalFrameRequest) -> dict:
        store = self._require_store()
        target_us = request.start_us
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
        path = self._out_dir / (
            f"{safe_name(request.participant_id)}_historical_{target_us}.png"
        )
        try:
            rgb = await asyncio.to_thread(nv12_to_rgb, frames[index], width, height)
            await asyncio.to_thread(save_png, rgb, path)
        except Exception as error:
            raise RPCError(f"Frame export failed: {error}", code="frame_export_error") from error
        return {
            "image": {"uri": str(path)},
            "width": width,
            "height": height,
            "timestamp_us": timestamp_us,
        }

    async def _sample_frames(
        self,
        request: LatestFramesRequest | HistoricalFramesRequest,
        start_us: int,
        end_us: int,
    ) -> dict:
        store = self._require_store()
        chunks = await asyncio.to_thread(
            store.overlapping_chunks,
            request.participant_id,
            start_us,
            end_us,
        )
        targets = sample_target_timestamps(start_us, end_us, request.frame_budget)

        assignments: dict[Path, list[int]] = {}
        metadata_by_path: dict[Path, dict] = {}
        for target_us in targets:
            path, metadata = min(
                chunks,
                key=lambda item: self._chunk_distance(item[0], item[1], target_us),
            )
            assignments.setdefault(path, []).append(target_us)
            metadata_by_path[path] = metadata

        sampled: list[dict] = []
        seen_timestamps: set[int] = set()
        for chunk, target_times in assignments.items():
            metadata = metadata_by_path[chunk]
            try:
                data = await asyncio.to_thread(chunk.read_bytes)
                frames = await asyncio.to_thread(decode_h264, data, self._gpu_id)
            except Exception as error:
                _LOGGER.warning("Skipping unavailable video chunk %s: %s", chunk, error)
                continue
            if not frames:
                _LOGGER.warning("Skipping video chunk %s: decoded zero frames", chunk)
                continue

            candidates = self._sample_candidates(
                chunk,
                metadata,
                len(frames),
                start_us,
                end_us,
            )
            if not candidates:
                continue
            selected = {
                min(candidates, key=lambda candidate: abs(candidate[1] - target_us))
                for target_us in target_times
            }
            for index, timestamp_us in sorted(selected, key=lambda candidate: candidate[1]):
                if timestamp_us in seen_timestamps:
                    continue
                seen_timestamps.add(timestamp_us)
                width = int(metadata.get("width", frames[index].shape[1]))
                height = int(metadata.get("height", frames[index].shape[0] * 2 // 3))
                resolution = (
                    "native"
                    if request.max_width is None
                    else f"{request.max_width}x{request.max_height}"
                )
                path = self._out_dir / (
                    f"{safe_name(request.participant_id)}_sample_"
                    f"{start_us}_{end_us}_{resolution}_{chunk.stem}_{index}.png"
                )
                try:
                    rgb = await asyncio.to_thread(
                        nv12_to_rgb,
                        frames[index],
                        width,
                        height,
                    )
                    width, height = await asyncio.to_thread(
                        save_png,
                        rgb,
                        path,
                        max_width=request.max_width,
                        max_height=request.max_height,
                    )
                except Exception as error:
                    raise RPCError(
                        f"Frame export failed: {error}",
                        code="frame_export_error",
                    ) from error
                sampled.append(
                    {
                        "image": {"uri": str(path)},
                        "width": width,
                        "height": height,
                        "timestamp_us": timestamp_us,
                    }
                )

        sampled.sort(key=lambda frame: frame["timestamp_us"])
        if not sampled:
            raise RPCError(
                "No decoded frames in requested time window",
                code="not_found",
            )
        return {
            "frames": sampled,
            "start_us": start_us,
            "end_us": end_us,
            "duration_seconds": request.duration_seconds,
            "frame_budget": request.frame_budget,
            "max_width": request.max_width,
            "max_height": request.max_height,
        }

    @staticmethod
    def _chunk_distance(path: Path, metadata: dict, target_us: int) -> tuple[int, int]:
        start_us, end_us = _chunk_bounds(path, metadata)
        distance = max(start_us - target_us, target_us - end_us, 0)
        midpoint_distance = abs(start_us + end_us - 2 * target_us)
        return distance, midpoint_distance

    @staticmethod
    def _sample_candidates(
        path: Path,
        metadata: dict,
        decoded_frames: int,
        window_start_us: int,
        window_end_us: int,
    ) -> list[tuple[int, int]]:
        start_us, end_us = _chunk_bounds(path, metadata)
        declared_frames = max(1, int(metadata.get("num_frames", decoded_frames)))
        usable_frames = min(decoded_frames, declared_frames)
        if declared_frames == 1 or end_us <= start_us:
            candidates = [(0, start_us)]
        else:
            candidates = [
                (
                    index,
                    start_us + index * (end_us - start_us) // (declared_frames - 1),
                )
                for index in range(usable_frames)
            ]
        in_window = [
            candidate
            for candidate in candidates
            if window_start_us <= candidate[1] <= window_end_us
        ]
        return in_window
