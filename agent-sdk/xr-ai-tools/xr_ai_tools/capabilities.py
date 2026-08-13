# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed native tools for shared XR capability services and text memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from xr_ai_models import VLMService

from .rpc import RPCClient
from .tools import Tool


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class SpatialFrame(BaseModel):
    origin: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3


class EmptyRequest(StrictRequest):
    pass


class HeadPose(BaseModel):
    is_valid: bool
    position: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3
    yaw_deg: float
    pitch_deg: float
    ts: int
    error: str | None = None


class TrackingTools:
    """Own the OpenXR-service client and current-user-frame tool."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)
        self.get_user_frame = Tool(
            "get_user_frame",
            "Get the user's current world-space origin and forward, right, and up axes.",
            EmptyRequest,
            SpatialFrame,
            self._get_user_frame,
        )

    async def _get_user_frame(self, request: EmptyRequest) -> SpatialFrame:
        pose = HeadPose.model_validate(
            await self._rpc.call("get_head_pose", request.model_dump())
        )
        if not pose.is_valid:
            raise RuntimeError(pose.error or "XR tracking is unavailable")
        return SpatialFrame(
            origin=pose.position,
            forward=pose.forward,
            right=pose.right,
            up=pose.up,
        )

    async def close(self) -> None:
        await self._rpc.close()


class ListRecordedParticipantsResult(BaseModel):
    participants: list[str]


class VideoStatsRequest(StrictRequest):
    participant_id: str = Field(min_length=1)


class VideoStatsResult(BaseModel):
    participant_id: str
    num_chunks: int
    total_bytes: int
    avg_chunk_bytes: int
    earliest_us: int
    latest_us: int


class QueryVideoRequest(StrictRequest):
    participant_id: str = Field(min_length=1)
    start_us: int = Field(gt=0)
    end_us: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> QueryVideoRequest:
        if self.end_us <= self.start_us:
            raise ValueError("end_us must be greater than start_us")
        return self


class QueryVideoResult(BaseModel):
    path: str
    size: int
    start_us: int
    end_us: int


class HistoricalFrameRequest(StrictRequest):
    participant_id: str = Field(min_length=1)
    second_ago: int = Field(default=0, ge=0)
    reference_time_us: int = Field(gt=0)


class HistoricalFrameResult(BaseModel):
    path: str
    width: int
    height: int
    timestamp_us: int
    second_ago: int
    actual_second_ago: float


class VideoMemoryTools:
    """Own a video-memory service client and its finite tools."""

    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)
        self.list_recorded_participants = Tool(
            "list_recorded_participants",
            "List exact participant identities with persisted camera history.",
            EmptyRequest,
            ListRecordedParticipantsResult,
            self._list_recorded_participants,
        )
        self.get_video_stats = Tool(
            "get_video_stats",
            "Return a participant's recorded Unix-epoch microsecond range and storage statistics.",
            VideoStatsRequest,
            VideoStatsResult,
            self._get_video_stats,
        )
        self.query_video = Tool(
            "query_video",
            "Write an H.264 clip overlapping an absolute Unix-epoch microsecond window and return its local path.",
            QueryVideoRequest,
            QueryVideoResult,
            self._query_video,
        )
        self.get_frame_from_time = Tool(
            "get_frame_from_time",
            "Extract the recorded PNG frame nearest reference_time_us minus second_ago whole seconds.",
            HistoricalFrameRequest,
            HistoricalFrameResult,
            self._get_frame_from_time,
        )
        self.tools = (
            self.list_recorded_participants,
            self.get_video_stats,
            self.query_video,
            self.get_frame_from_time,
        )

    async def _list_recorded_participants(
        self,
        request: EmptyRequest,
    ) -> ListRecordedParticipantsResult:
        return ListRecordedParticipantsResult.model_validate(
            await self._rpc.call("list_recorded_participants", request.model_dump())
        )

    async def _get_video_stats(self, request: VideoStatsRequest) -> VideoStatsResult:
        return VideoStatsResult.model_validate(
            await self._rpc.call("get_video_stats", request.model_dump())
        )

    async def _query_video(self, request: QueryVideoRequest) -> QueryVideoResult:
        return QueryVideoResult.model_validate(
            await self._rpc.call("query_video", request.model_dump())
        )

    async def _get_frame_from_time(
        self,
        request: HistoricalFrameRequest,
    ) -> HistoricalFrameResult:
        return HistoricalFrameResult.model_validate(
            await self._rpc.call("get_frame_from_time", request.model_dump())
        )

    async def close(self) -> None:
        await self._rpc.close()


class HistoricalVisionRequest(StrictRequest):
    participant_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    second_ago: int = Field(gt=0)
    reference_time_us: int = Field(gt=0)


class VisionResult(BaseModel):
    text: str


class HistoricalVisionTool(Tool[HistoricalVisionRequest, VisionResult]):
    """Answer one question from a frame in recorded video memory."""

    def __init__(self, *, video: VideoMemoryTools, vlm: VLMService) -> None:
        self.video = video
        self.vlm = vlm
        super().__init__(
            "look_at_past_frame",
            "Inspect a recorded camera frame for an explicitly historical question.",
            HistoricalVisionRequest,
            VisionResult,
            self._answer,
        )

    async def _answer(self, request: HistoricalVisionRequest) -> VisionResult:
        frame = await self.video.get_frame_from_time.execute(
            HistoricalFrameRequest(
                participant_id=request.participant_id,
                second_ago=request.second_ago,
                reference_time_us=request.reference_time_us,
            )
        )
        response = await self.vlm.ask_image(
            Path(frame.path),
            request.query,
            system_prompt="Answer directly from this recorded camera frame in one short sentence.",
        )
        text = (response.content or "").strip()
        if not text:
            raise RuntimeError("The recorded camera image did not produce an answer.")
        return VisionResult(text=text)


class AddTranscriptRequest(StrictRequest):
    source_id: str
    timestamp_us: int
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AddTranscriptResult(BaseModel):
    ok: bool = True


class TextMemoryTool(Tool[AddTranscriptRequest, AddTranscriptResult]):
    """Append timestamped text to participant-scoped JSONL storage."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._root = self.directory.resolve()
        self._lock = Lock()
        super().__init__(
            "add_transcript",
            "Append one timestamped text segment to persistent memory.",
            AddTranscriptRequest,
            AddTranscriptResult,
            self._append,
        )

    @staticmethod
    def _safe(source_id: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in source_id
        )

    def _path(self, source_id: str) -> Path:
        stem = self._safe(source_id)
        suffix = 1
        while True:
            candidate = stem if suffix == 1 else f"{stem}_{suffix}"
            identity = (self.directory / f"{candidate}.identity").resolve()
            data = (self.directory / f"{candidate}.jsonl").resolve()
            if not identity.is_relative_to(self._root) or not data.is_relative_to(self._root):
                raise ValueError("transcript path escapes storage directory")
            if identity.exists() and identity.read_text(encoding="utf-8") == source_id:
                return data
            if suffix == 1 and data.exists() and not identity.exists() and source_id == stem:
                identity.write_text(source_id, encoding="utf-8")
                return data
            if not identity.exists() and not data.exists():
                identity.write_text(source_id, encoding="utf-8")
                return data
            suffix += 1

    async def _append(self, request: AddTranscriptRequest) -> AddTranscriptResult:
        await asyncio.to_thread(self._append_sync, request)
        return AddTranscriptResult()

    def _append_sync(self, request: AddTranscriptRequest) -> None:
        with self._lock:
            with self._path(request.source_id).open("a", encoding="utf-8") as output:
                output.write(
                    json.dumps(
                        {
                            "timestamp_us": request.timestamp_us,
                            "text": request.text,
                        }
                    )
                    + "\n"
                )


__all__ = [
    "AddTranscriptRequest",
    "EmptyRequest",
    "HistoricalVisionRequest",
    "HistoricalVisionTool",
    "SpatialFrame",
    "TextMemoryTool",
    "TrackingTools",
    "Vector3",
    "VideoMemoryTools",
]
