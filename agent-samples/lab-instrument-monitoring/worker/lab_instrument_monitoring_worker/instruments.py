# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Marker-associated lab-instrument reading."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import VLMService
from xr_ai_runtime import Agent
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.marker_tracking import (
    MarkerTrackingRequest,
    TrackedMarker,
)
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from .device_map import DeviceMap
from .events import InstrumentReading, InstrumentSighting
from .images import ParticipantImageAgent


class ReadLabInstrumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)


class LabInstrumentReadResult(BaseModel):
    readings: list[InstrumentReading] = Field(default_factory=list)
    sightings: list[InstrumentSighting] = Field(default_factory=list)
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
        prompt: str,
        debug_dir: Path | None = None,
    ) -> None:
        self._images = images
        self._device_map = device_map
        self._debug_dir = debug_dir
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        self._query_image = ImageQueryTool(
            images=images.images,
            vlm=vlm,
            system_prompt=prompt,
        )
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

        labeled_markers = [
            (f"M{index}", marker)
            for index, marker in enumerate(
                sorted(markers, key=_marker_position),
                start=1,
            )
        ]
        labels = [label for label, _marker in labeled_markers]
        mapped: dict[str, tuple[TrackedMarker, str]] = {}
        sightings: list[InstrumentSighting] = []
        for label, marker in labeled_markers:
            identity = self._device_map.resolve(marker.marker_type, marker.value)
            if identity is None:
                logger.warning(
                    "ignoring unmapped instrument marker marker={}",
                    _marker_log_id(marker),
                )
                continue
            mapped[label] = (marker, identity.device_name)
            sightings.append(
                InstrumentSighting(
                    timestamp_us=frame.timestamp_us,
                    marker_type=marker.marker_type,
                    marker_id=marker.value,
                    device_name=identity.device_name,
                )
            )

        if not mapped:
            return LabInstrumentReadResult(
                sightings=sightings,
                available=False,
                message="No configured marker-labelled lab instruments were found.",
            )

        result = None
        try:
            annotated_bytes = await asyncio.to_thread(
                _annotate_markers,
                source,
                labeled_markers,
            )
            annotated = self._images.images.put_derived(
                annotated_bytes,
                source=frame.image,
            )
            result = await self._query_image.execute(
                ImageQueryRequest(
                    image=annotated,
                    query=self._reading_query(labels),
                )
            )
            parsed = _parse_joint_readings(result.text, labels)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "joint instrument display read failed pid={!r}",
                request.participant_id,
            )
            parsed = None

        if result is None or not result.available or parsed is None:
            return LabInstrumentReadResult(
                sightings=sightings,
                available=False,
                message="Markers were found, but the instrument display response was invalid.",
            )

        readings = [
            InstrumentReading(
                timestamp_us=frame.timestamp_us,
                marker_type=marker.marker_type,
                marker_id=marker.value,
                device_name=device_name,
                meter_reading=parsed[label],
            )
            for label, (marker, device_name) in mapped.items()
            if parsed[label].upper() != "UNKNOWN"
        ]
        if not readings:
            return LabInstrumentReadResult(
                sightings=sightings,
                available=False,
                message="Markers were found, but their instrument displays could not be read.",
            )
        return LabInstrumentReadResult(readings=readings, sightings=sightings)

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

    @staticmethod
    def _reading_query(labels: list[str]) -> str:
        keys = json.dumps(labels)
        return (
            "Read all marker-labelled instruments together. Return one JSON object with exactly "
            f"these keys: {keys}. Each value must be the reading and unit from that marker's own "
            "physical instrument, or UNKNOWN. Never assign one display to multiple markers."
        )

    @staticmethod
    def render_readings(result: LabInstrumentReadResult) -> str:
        if not result.readings:
            return result.message or "No lab instrument readings were available."
        return "; ".join(f"{reading.device_name}: {reading.meter_reading}" for reading in result.readings)


def _marker_log_id(marker: TrackedMarker) -> str:
    digest = hashlib.sha256(marker.value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{marker.marker_type.value}:{digest}"


def _marker_position(marker: TrackedMarker) -> tuple[float, float]:
    return (
        sum(point.y for point in marker.corners) / len(marker.corners),
        sum(point.x for point in marker.corners) / len(marker.corners),
    )


def _annotate_markers(
    source: bytes,
    labeled_markers: list[tuple[str, TrackedMarker]],
) -> bytes:
    palette = (
        (255, 0, 255),
        (0, 255, 255),
        (255, 180, 0),
        (80, 220, 80),
        (255, 90, 90),
        (100, 160, 255),
    )
    with Image.open(io.BytesIO(source)) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, (label, marker) in enumerate(labeled_markers):
        points = [(point.x, point.y) for point in marker.corners]
        draw.polygon(points, fill=palette[index % len(palette)])
        left = min(point[0] for point in points)
        right = max(point[0] for point in points)
        top = min(point[1] for point in points)
        bottom = max(point[1] for point in points)
        font_size = max(14, min(42, int(min(right - left, bottom - top) * 0.55)))
        font = ImageFont.load_default(size=font_size)
        draw.text(
            ((left + right) / 2, (top + bottom) / 2),
            label,
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _parse_joint_readings(text: str, labels: list[str]) -> dict[str, str] | None:
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = visible.find("{")
    end = visible.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(visible[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != set(labels):
        return None
    if not all(isinstance(value, str) and value.strip() for value in payload.values()):
        return None
    return {label: payload[label].strip() for label in labels}


__all__ = [
    "LabInstrumentReadResult",
    "LabInstrumentAgent",
    "ReadLabInstrumentsRequest",
]
