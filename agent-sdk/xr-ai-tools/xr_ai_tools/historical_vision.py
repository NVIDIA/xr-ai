# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native visual question answering over recorded video frames."""

from pathlib import Path

from pydantic import BaseModel, Field
from xr_ai_models import VLMService

from .tools import Tool
from .types import StrictRequest
from .video_memory import HistoricalFrameRequest, VideoMemoryTools


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


__all__ = ["HistoricalVisionRequest", "HistoricalVisionTool", "VisionResult"]
