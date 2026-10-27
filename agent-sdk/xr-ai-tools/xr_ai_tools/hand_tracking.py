# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mono-image hand landmarks for selected images; no gesture or action policy."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .image import ImageInput, ImageReference, ImageRegistry
from .tools import Tool
from .types import StrictRequest


class HandLandmark(BaseModel):
    """One of the 21 MediaPipe-ordered landmarks in source-image coordinates."""

    x: float = Field(allow_inf_nan=False)
    """Horizontal fraction of source width; predictions can extend outside the image."""
    y: float = Field(allow_inf_nan=False)
    """Vertical fraction of source height, increasing downwards."""
    z: float = Field(allow_inf_nan=False)
    """Relative wrist depth, scaled approximately like x; not metric depth."""


class TrackedHand(BaseModel):
    """Hand landmarks and the model's handedness classification."""

    landmarks: list[HandLandmark] = Field(min_length=21, max_length=21)
    """Landmarks in MediaPipe order: wrist 0, thumb tip 4, index tip 8."""
    handedness: str
    """Model-reported Left or Right; not a stable hand identity."""
    handedness_score: float = Field(ge=0, le=1)
    """Confidence in handedness, not a hand-detection confidence."""


class HandTrackingRequest(StrictRequest):
    """Detect hands in an already selected image."""

    image: ImageReference
    """Image selected by a frame, still, or other image tool."""


class HandTrackingResult(BaseModel):
    """Hand observations in one source image."""

    hands: list[TrackedHand]
    """Detected hands; empty when no hand is found."""
    width: int = Field(gt=0)
    """Source image width in pixels."""
    height: int = Field(gt=0)
    """Source image height in pixels."""


class HandTrackingTool(Tool[HandTrackingRequest, HandTrackingResult]):
    """Detect mono RGB hand landmarks using an optional local MediaPipe backend.

    Install ``hand-tracking`` and provide a locally provisioned Hand Landmarker
    task bundle. No model is downloaded automatically. Independent IMAGE-mode
    calls avoid sharing temporal state between participants. The tool owns one
    detector; close it after use. Gestures and image acquisition belong to callers.
    """

    def __init__(self, *, images: ImageRegistry, model_path: Path, num_hands: int = 2) -> None:
        if num_hands < 1:
            raise ValueError("num_hands must be positive")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        self._images = images
        self._lock = Lock()
        self._closed = False
        self._detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE, num_hands=num_hands,
        ))
        super().__init__("track_hands", "Detect hand landmarks in a selected image.",
                         HandTrackingRequest, HandTrackingResult, self._track)

    async def _track(self, request: HandTrackingRequest) -> HandTrackingResult:
        source = self._images.resolve(request.image)
        return await asyncio.to_thread(self._detect, source)

    def _detect(self, source: ImageInput) -> HandTrackingResult:
        # Cancellation cannot stop native inference. Close and every detector
        # access share this lock so teardown waits for actual native completion.
        with self._lock:
            if self._closed:
                raise RuntimeError("hand tracking is closed")
            if isinstance(source, str) and "://" in source:
                raise ValueError("hand tracking requires image bytes or a local path")
            with Image.open(io.BytesIO(source) if isinstance(source, bytes) else Path(source)) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
            result = self._detector.detect(self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb),
            ))
            hands = [
                TrackedHand(
                    landmarks=[HandLandmark(x=p.x, y=p.y, z=p.z) for p in points],
                    handedness=categories[0].category_name,
                    handedness_score=categories[0].score,
                )
                for points, categories in zip(result.hand_landmarks, result.handedness, strict=True)
            ]
            return HandTrackingResult(hands=hands, width=rgb.shape[1], height=rgb.shape[0])

    def close(self) -> None:
        """Wait for native detection and close; call off the event loop during async use."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._detector.close()


__all__ = ["HandLandmark", "TrackedHand", "HandTrackingRequest", "HandTrackingResult", "HandTrackingTool"]
