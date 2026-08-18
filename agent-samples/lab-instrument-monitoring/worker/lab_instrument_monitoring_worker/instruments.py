# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Marker-associated lab-instrument reading."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import VLMService
from xr_ai_runtime import Agent
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.image import ImageReference
from xr_ai_tools.image_polygon import (
    ImagePoint,
    ImagePolygonFillRequest,
    ImagePolygonFillTool,
)
from xr_ai_tools.marker_tracking import (
    MarkerTrackingRequest,
    TrackedMarker,
)
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from .device_map import DeviceMap
from .events import InstrumentReading
from .images import ParticipantImageAgent


class ReadLabInstrumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)


class LabInstrumentReadResult(BaseModel):
    readings: list[InstrumentReading] = Field(default_factory=list)
    available: bool = True
    message: str = ""


class LabInstrumentAgent(Agent):
    """Read marker-identified instruments from one current frame."""

    def __init__(
        self,
        *,
        images: ParticipantImageAgent,
        vlm: VLMService,
        device_map: DeviceMap,
        debug_dir: Path | None = None,
    ) -> None:
        self._images = images
        self._device_map = device_map
        self._debug_dir = debug_dir
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        self._query_image = ImageQueryTool(images=images.images, vlm=vlm)
        self._fill_polygon = ImagePolygonFillTool(images=images.images)
        self.read_lab_instruments = Tool(
            "read_lab_instruments",
            "Read every visible lab instrument display and associate each reading with its configured marker identity.",
            ReadLabInstrumentsRequest,
            LabInstrumentReadResult,
            self._read_lab_instruments,
            render_result=self.render_readings,
        )
        super().__init__((self.read_lab_instruments,))

    async def _read_lab_instruments(
        self,
        request: ReadLabInstrumentsRequest,
    ) -> LabInstrumentReadResult:
        scan_path: Path | None = None
        try:
            frame = await self._images.get_current_frame.execute(
                CurrentFrameRequest(participant_id=request.participant_id)
            )
            source = self._images.images.resolve(frame.image)
            if not isinstance(source, bytes):
                raise TypeError("current camera image must resolve to bytes")
            scan_path = await self._record_scan_image(
                request.participant_id,
                frame.timestamp_us,
                frame.sequence,
                source,
            )
            tracked = await self._images.track_markers.execute(
                MarkerTrackingRequest(
                    participant_id=request.participant_id,
                    image=frame.image,
                )
            )
            if not tracked.available:
                return LabInstrumentReadResult(
                    available=False,
                    message=tracked.message or "The camera frame could not be scanned.",
                )
            markers = tracked.markers
            marker_families = Counter(marker.marker_type.value for marker in markers)
            logger.info(
                "instrument marker scan pid={!r} image={} marker_families={}",
                request.participant_id,
                scan_path,
                dict(marker_families),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).warning(
                "instrument frame or marker scan failed pid={!r}",
                request.participant_id,
            )
            return LabInstrumentReadResult(available=False, message=str(exc))

        if not markers:
            return LabInstrumentReadResult(message="No readable marker-labelled lab instruments were found.")

        readings: list[InstrumentReading] = []
        for marker in markers:
            try:
                reading = await self._read_one(
                    frame.image,
                    frame.timestamp_us,
                    marker,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning(
                    "instrument display read failed pid={!r} marker={}",
                    request.participant_id,
                    _marker_log_id(marker),
                )
                continue
            if reading is not None:
                readings.append(reading)
        if not readings:
            return LabInstrumentReadResult(
                available=False,
                message="Markers were found, but their instrument displays could not be read.",
            )
        return LabInstrumentReadResult(readings=readings)

    async def _record_scan_image(
        self,
        participant_id: str,
        frame_timestamp_us: int,
        sequence: int,
        image: bytes,
    ) -> Path | None:
        if self._debug_dir is None:
            return None
        safe_participant = "".join(
            character if character.isalnum() or character in "-_." else "-" for character in participant_id
        )
        invoked_at_us = time.time_ns() // 1_000
        path = self._debug_dir / (f"{invoked_at_us}-{safe_participant}-frame-{frame_timestamp_us}-seq-{sequence}.jpg")
        await asyncio.to_thread(path.write_bytes, image)
        return path

    async def _read_one(
        self,
        image: ImageReference,
        timestamp_us: int,
        marker: TrackedMarker,
    ) -> InstrumentReading | None:
        identity = self._device_map.resolve(marker.marker_type, marker.value)
        if identity is None:
            logger.warning(
                "ignoring unmapped instrument marker marker={}",
                _marker_log_id(marker),
            )
            return None
        marked = await self._fill_polygon.execute(
            ImagePolygonFillRequest(
                image=image,
                coordinates=[ImagePoint(x=point.x, y=point.y) for point in marker.corners],
            )
        )
        if not marked.available or marked.image is None:
            return None
        result = await self._query_image.execute(
            ImageQueryRequest(
                image=marked.image,
                query=self._reading_query(marker, identity.device_name),
            )
        )
        reading = result.text.strip()
        if not result.available or not reading or reading.upper() == "UNKNOWN":
            return None
        return InstrumentReading(
            timestamp_us=timestamp_us,
            marker_type=marker.marker_type,
            marker_id=marker.value,
            device_name=identity.device_name,
            meter_reading=reading,
        )

    @staticmethod
    def _reading_query(marker: TrackedMarker, device_name: str) -> str:
        return (
            "The magenta polygon marks the "
            f"{marker.marker_type.value} marker {marker.value!r} attached to the lab "
            f"instrument named {device_name!r}. Read only the display physically on the same "
            "instrument body as the highlighted marker. Do not use a reading from an adjacent, "
            "overlapping, or nearby instrument. Return only the displayed meter reading including "
            "its unit. Return UNKNOWN if the highlighted marker and display are not clearly part "
            "of the same device, or if the reading is unreadable."
        )

    @staticmethod
    def render_readings(result: LabInstrumentReadResult) -> str:
        if not result.readings:
            return result.message or "No lab instrument readings were available."
        return "; ".join(f"{reading.device_name}: {reading.meter_reading}" for reading in result.readings)


def _marker_log_id(marker: TrackedMarker) -> str:
    digest = hashlib.sha256(marker.value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{marker.marker_type.value}:{digest}"


__all__ = [
    "LabInstrumentReadResult",
    "LabInstrumentAgent",
    "ReadLabInstrumentsRequest",
]
