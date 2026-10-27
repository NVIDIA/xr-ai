# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose independent tools into gesture → photograph → crop → OCR."""

from __future__ import annotations

from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.hand_tracking import HandTrackingRequest, HandTrackingTool
from xr_ai_tools.image_crop import ImageCropRequest, ImageCropTool
from xr_ai_tools.still_capture import StillCaptureRequest, StillCaptureTool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from .config import Settings
from .gesture import CameraBoxTrigger, camera_box


class GestureCaptureAgent:
    """One participant's workflow; the application owns polling and task lifetime."""

    def __init__(
        self, *, frames: CurrentFrameTool, hands: HandTrackingTool,
        capture: StillCaptureTool, crop: ImageCropTool, query: ImageQueryTool,
        trigger: CameraBoxTrigger, settings: Settings,
    ) -> None:
        self._frames = frames
        self._hands = hands
        self._capture = capture
        self._crop = crop
        self._query = query
        self._trigger = trigger
        self._settings = settings
        self._last_frame = None

    async def step(self, participant_id: str) -> str | None:
        frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id, allow_capture=False))
        identity = (frame.track_id, frame.sequence, frame.timestamp_us)
        if identity == self._last_frame:
            return None
        if self._last_frame is not None and self._last_frame[0] != frame.track_id:
            self._trigger.unavailable()
        self._last_frame = identity
        hands = await self._hands.execute(HandTrackingRequest(image=frame.image))
        region = camera_box(hands, corner_tolerance=self._settings.corner_tolerance,
                            angle_cosine=self._settings.angle_cosine)
        region = self._trigger.observe(region, frame.timestamp_us)
        if region is None:
            return None
        photo = await self._capture.execute(StillCaptureRequest(participant_id=participant_id))
        if photo.width * frame.height != photo.height * frame.width:
            raise ValueError("Photo and video aspect ratios differ; crop alignment is unavailable")
        cropped = await self._crop.execute(ImageCropRequest(image=photo.image, region=region))
        answer = await self._query.execute(ImageQueryRequest(image=cropped.image, query=self._settings.query))
        return answer.text

    def unavailable(self) -> None:
        self._trigger.unavailable()
