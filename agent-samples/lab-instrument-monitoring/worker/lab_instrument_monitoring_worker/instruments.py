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
from PIL import Image, ImageDraw
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

RGBColor = tuple[int, int, int]
ColoredMarker = tuple[str, RGBColor, TrackedMarker]

_MARKER_COLORS: tuple[tuple[str, RGBColor], ...] = (
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("orange", (255, 180, 0)),
    ("green", (80, 220, 80)),
    ("red", (255, 90, 90)),
    ("blue", (100, 160, 255)),
)


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

        try:
            colored_markers = _assign_marker_colors(markers)
        except ValueError as exc:
            return LabInstrumentReadResult(available=False, message=str(exc))
        color_keys = [color_name for color_name, _color, _marker in colored_markers]
        mapped: dict[str, tuple[TrackedMarker, str]] = {}
        sightings: list[InstrumentSighting] = []
        for color_name, _color, marker in colored_markers:
            identity = self._device_map.resolve(marker.marker_type, marker.value)
            if identity is None:
                logger.warning(
                    "ignoring unmapped instrument marker marker={}",
                    _marker_log_id(marker),
                )
                continue
            mapped[color_name] = (marker, identity.device_name)
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
                colored_markers,
            )
            annotated = self._images.images.put_derived(
                annotated_bytes,
                source=frame.image,
            )
            result = await self._query_image.execute(
                ImageQueryRequest(
                    image=annotated,
                    query=self._reading_query(color_keys),
                )
            )
            parsed = _parse_joint_readings(result.text, color_keys) if result.available else None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "joint instrument display read failed pid={!r}",
                request.participant_id,
            )
            parsed = None

        if result is None:
            return LabInstrumentReadResult(
                sightings=sightings,
                available=False,
                message="Markers were found, but the instrument display response was invalid.",
            )
        if not result.available:
            return LabInstrumentReadResult(
                sightings=sightings,
                available=False,
                message=result.text.strip() or "The instrument vision model was unavailable.",
            )
        if parsed is None:
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
                meter_reading=parsed[color_name],
            )
            for color_name, (marker, device_name) in mapped.items()
            if parsed[color_name].upper() != "UNKNOWN"
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
    def _reading_query(color_keys: list[str]) -> str:
        keys = json.dumps(color_keys)
        return (
            "Read all color-block-marked instruments together. The solid color blocks are the "
            f"device identifiers: {keys}. Return one JSON object with exactly those color names "
            "as keys. Each value must be the reading and unit from that color block's own physical "
            "instrument, or UNKNOWN. Never assign one display to multiple color blocks."
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


def _assign_marker_colors(markers: list[TrackedMarker]) -> list[ColoredMarker]:
    if len(markers) > len(_MARKER_COLORS):
        raise ValueError(
            f"Found {len(markers)} markers, but only {len(_MARKER_COLORS)} unique marker colors are configured."
        )
    return [
        (color_name, color, marker)
        for (color_name, color), marker in zip(
            _MARKER_COLORS,
            sorted(markers, key=_marker_position),
            strict=False,
        )
    ]


def _annotate_markers(
    source: bytes,
    colored_markers: list[ColoredMarker],
) -> bytes:
    with Image.open(io.BytesIO(source)) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for _color_name, color, marker in colored_markers:
        points = [(point.x, point.y) for point in marker.corners]
        draw.polygon(points, fill=color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _parse_joint_readings(text: str, color_keys: list[str]) -> dict[str, str] | None:
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = visible.find("{")
    end = visible.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(visible[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != set(color_keys):
        return None
    if not all(isinstance(value, str) and value.strip() for value in payload.values()):
        return None
    return {color_name: payload[color_name].strip() for color_name in color_keys}


__all__ = [
    "LabInstrumentReadResult",
    "LabInstrumentAgent",
    "ReadLabInstrumentsRequest",
]
