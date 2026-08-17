# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select current camera frames without invoking a VLM."""

from __future__ import annotations

import asyncio

from pydantic import Field
from xr_ai_hub import LiveFrameSource, ProcessorEndpoint

from ._pixels import encode_image_bytes, frame_to_pil
from .image import ImageRegistry, TimedImage
from .tools import Tool
from .types import StrictRequest


class ImageFrame(TimedImage):
    """One selected camera frame and its source metadata."""

    width: int = Field(gt=0)
    """Frame width in pixels."""

    height: int = Field(gt=0)
    """Frame height in pixels."""

    sequence: int = Field(ge=0)
    """Source track sequence number."""

    participant_id: str = Field(min_length=1)
    """Participant that published the frame."""

    track_id: str = ""
    """Source video track identifier, when available."""


class CurrentFrameRequest(StrictRequest):
    """Select the latest camera frame for one participant."""

    participant_id: str = Field(
        min_length=1,
        description="Participant whose latest camera frame should be returned.",
    )
    """Participant whose latest camera frame should be returned."""


class CurrentFrameTool(Tool[CurrentFrameRequest, ImageFrame]):
    """Return a participant's current frame without running visual inference."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        images: ImageRegistry,
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        if frame_max_age_s <= 0.0:
            raise ValueError("frame_max_age_s must be positive")
        if frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")
        self.images = images
        self.frames = LiveFrameSource(
            endpoint,
            max_age_s=frame_max_age_s,
            timeout_s=frame_timeout_s,
        )
        super().__init__(
            "get_current_frame",
            "Return a participant's latest available camera frame without interpreting it.",
            CurrentFrameRequest,
            ImageFrame,
            self._get_current_frame,
        )

    def release(self, participant_id: str) -> None:
        """Forget cached frame state and image handles after disconnect."""

        self.frames.release(participant_id)
        self.images.release_owner(participant_id)

    async def _get_current_frame(self, request: CurrentFrameRequest) -> ImageFrame:
        frame = await self.frames.get(request.participant_id)
        image_bytes = await asyncio.to_thread(lambda: encode_image_bytes(frame_to_pil(frame)))
        return ImageFrame(
            image=self.images.put(image_bytes, owner=request.participant_id),
            width=frame.width,
            height=frame.height,
            timestamp_us=frame.pts_us,
            sequence=frame.seq,
            participant_id=frame.participant_id or request.participant_id,
            track_id=frame.track_id,
        )


__all__ = ["CurrentFrameRequest", "CurrentFrameTool", "ImageFrame"]
