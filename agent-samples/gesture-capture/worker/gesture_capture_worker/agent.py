# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose independent tools into gesture → selected image → crop → VLM."""

from __future__ import annotations

import time

from xr_ai_hub import FrameUnavailable
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.hand_tracking import HandTrackingRequest, HandTrackingTool
from xr_ai_tools.image_crop import ImageCropRequest, ImageCropTool
from xr_ai_tools.still_capture import StillCaptureRequest, StillCaptureTool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from .config import Settings
from .diagnostics import GestureDiagnostics
from .feedback import GestureFeedback
from .gesture import RegionSelector
from .selection_wait import CircleCaptureWait


class GestureCaptureAgent:
    """One participant's workflow; the application owns polling and task lifetime."""

    def __init__(
        self, *, frames: CurrentFrameTool, hands: HandTrackingTool,
        capture: StillCaptureTool, crop: ImageCropTool, query: ImageQueryTool,
        selector: RegionSelector, settings: Settings, feedback: GestureFeedback | None = None,
        diagnostics: GestureDiagnostics | None = None,
    ) -> None:
        self._diagnostics = diagnostics or GestureDiagnostics()
        self._frames = frames
        self._hands = hands
        self._capture = capture
        self._crop = crop
        self._query = query
        self._selector = selector
        self._settings = settings
        self._last_frame = None
        self._feedback = feedback
        self._turn_active = False
        self._pending = None

    async def step(self, participant_id: str) -> str | None:
        try:
            return await self._step(participant_id)
        except FrameUnavailable:
            self._diagnostics.count("frame_unavailable")
            self.unavailable()
            raise
        except BaseException:
            self._diagnostics.count("step_failed_or_cancelled")
            self._pending = None
            raise
        finally:
            self._diagnostics.emit()

    async def _step(self, participant_id: str) -> str | None:
        if self._pending is not None and self._pending.expired:
            self._diagnostics.count("pending_no_fresh_video")
            self._pending = None
            raise ValueError("Circle selection expired without fresh video; draw again")
        if self._turn_active and self._pending is None:
            return None
        frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id, allow_capture=False))
        identity = (frame.track_id, frame.sequence, frame.timestamp_us)
        if identity == self._last_frame:
            self._diagnostics.count("duplicate_frame")
            return None
        if self._last_frame is not None and self._last_frame[0] != frame.track_id:
            self._diagnostics.count("source_changed")
            self._selector.reset()
        self._last_frame = identity
        started = time.monotonic()
        hands = await self._hands.execute(HandTrackingRequest(image=frame.image))
        if self._diagnostics.enabled:
            self._diagnostics.frame(frame.timestamp_us, len(hands.hands), (time.monotonic() - started) * 1000)
        if self._pending is not None:
            ready = self._pending.observe(hands, frame)
            if not ready:
                return None
            region = self._pending.region
            self._pending = None
        else:
            region = self._selector.observe(hands, frame.timestamp_us)
            if region is None:
                if self._feedback:
                    await self._feedback.update(self._selector.in_progress)
                return None
            self._turn_active = True
            if self._feedback:
                await self._feedback.recognized()
            if self._selector.circle_selected:
                self._pending = CircleCaptureWait(
                    region, self._selector.drawing_hand, frame, self._settings, self._diagnostics)
                return None
        source = frame
        if self._settings.aligned_photo:
            source = await self._capture.execute(StillCaptureRequest(participant_id=participant_id))
            if source.width * frame.height != source.height * frame.width:
                raise ValueError("Photo and video aspect ratios differ; crop alignment is unavailable")
        cropped = await self._crop.execute(ImageCropRequest(image=source.image, region=region))
        answer = await self._query.execute(ImageQueryRequest(image=cropped.image, query=self._settings.query))
        return answer.text

    def unavailable(self) -> None:
        if self._pending is not None:
            self._pending.clear_since = None
            if self._pending.expired:
                self._pending = None
        self._selector.unavailable()

    async def finish_turn(self) -> None:
        """Pending clearance continues observing; completed turns unlock after delivery."""
        if self._pending is None:
            self._turn_active = False
