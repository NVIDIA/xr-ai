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
from PIL import Image, ImageChops, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import VLMService
from xr_ai_runtime import Agent
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.marker_tracking import (
    MarkerTrackingRequest,
    MarkerType,
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
    ("red", (255, 0, 0)),
    ("green", (0, 128, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
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

        mapped_markers: list[TrackedMarker] = []
        device_names: dict[tuple[MarkerType, str], str] = {}
        sightings: list[InstrumentSighting] = []
        for marker in markers:
            identity = self._device_map.resolve(marker.marker_type, marker.value)
            if identity is None:
                logger.warning(
                    "ignoring unmapped instrument marker marker={}",
                    _marker_log_id(marker),
                )
                continue
            mapped_markers.append(marker)
            device_names[(marker.marker_type, marker.value)] = identity.device_name
            sightings.append(
                InstrumentSighting(
                    timestamp_us=frame.timestamp_us,
                    marker_type=marker.marker_type,
                    marker_id=marker.value,
                    device_name=identity.device_name,
                )
            )

        if not mapped_markers:
            return LabInstrumentReadResult(
                sightings=sightings,
                available=False,
                message="No configured marker-labelled lab instruments were found.",
            )

        if len(mapped_markers) > len(_MARKER_COLORS):
            logger.warning(
                "instrument marker palette exhausted pid={!r} mapped={} supported={} ignored={}",
                request.participant_id,
                len(mapped_markers),
                len(_MARKER_COLORS),
                len(mapped_markers) - len(_MARKER_COLORS),
            )
        colored_markers = _assign_marker_colors(mapped_markers)
        color_keys = [color_name for color_name, _color, _marker in colored_markers]
        mapped = {
            color_name: (
                marker,
                device_names[(marker.marker_type, marker.value)],
            )
            for color_name, _color, marker in colored_markers
        }

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
        response_template = json.dumps(
            {color_name: {"reading": "UNKNOWN", "display_bbox": None} for color_name in color_keys}
        )
        return (
            f"The requested colored-X keys are: {keys}. "
            "Apply this association rule to every key: a reading belongs to a color only when "
            "that colored X and the display are on the same visually bounded instrument. "
            "For each color, directly return the associated display reading and its normalized "
            "[left, top, right, bottom] bounding box using coordinates from 0 through 1000. "
            "Use UNKNOWN and null when that color has no unambiguous readable display. Never use "
            "the same physical display for more than one color; separate displays may show the "
            "same reading. "
            f"Return exactly this JSON shape, replacing only its values: {response_template}. "
            "No explanation or Markdown."
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
    for _color_name, color, marker in colored_markers:
        points = [(point.x, point.y) for point in marker.corners]
        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)
        horizontal_span = right - left
        vertical_span = bottom - top
        stroke_width = max(3, round(min(horizontal_span, vertical_span) * 0.24))

        marker_mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(marker_mask).polygon(points, fill=255)
        x_mask = Image.new("L", image.size, 0)
        x_draw = ImageDraw.Draw(x_mask)
        x_draw.line((left, top, right, bottom), fill=255, width=stroke_width)
        x_draw.line((right, top, left, bottom), fill=255, width=stroke_width)
        image.paste(color, mask=ImageChops.multiply(marker_mask, x_mask))
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
        pairs = json.loads(
            visible[start : end + 1],
            object_pairs_hook=list,
        )
    except json.JSONDecodeError:
        return None
    payload_result = _unique_object(pairs)
    if payload_result is None:
        return None
    payload, duplicate_colors = payload_result

    expected_keys = {color_name: color_name.strip().lower() for color_name in color_keys}
    if len(set(expected_keys.values())) != len(color_keys):
        return None

    readings = dict.fromkeys(color_keys, "UNKNOWN")
    display_regions: dict[str, tuple[float, float, float, float]] = {}
    for color_name, normalized_name in expected_keys.items():
        if normalized_name in duplicate_colors:
            continue
        result = _unique_object(payload.get(normalized_name))
        if result is None:
            continue
        value, duplicate_fields = result
        if duplicate_fields.intersection({"reading", "display_bbox"}):
            continue
        raw_reading = value.get("reading")
        if isinstance(raw_reading, str) and raw_reading.strip().upper() == "UNKNOWN":
            continue
        reading = _valid_reading(raw_reading)
        bbox = _valid_bbox(value.get("display_bbox"))
        if reading is None or bbox is None:
            continue
        readings[color_name] = reading
        display_regions[color_name] = bbox

    conflicts: set[str] = set()
    assigned = list(display_regions.items())
    for index, (first_color, first_bbox) in enumerate(assigned):
        for second_color, second_bbox in assigned[index + 1 :]:
            if _same_display_region(first_bbox, second_bbox):
                conflicts.update((first_color, second_color))

    for color_name in conflicts:
        readings[color_name] = "UNKNOWN"
    return readings


def _unique_object(value: object) -> tuple[dict[str, object], set[str]] | None:
    if not isinstance(value, list) or not all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) for pair in value
    ):
        return None
    payload: dict[str, object] = {}
    duplicate_keys: set[str] = set()
    for key, item in value:
        normalized_key = key.strip().lower()
        if normalized_key in payload:
            duplicate_keys.add(normalized_key)
        payload[normalized_key] = item
    return payload, duplicate_keys


def _valid_reading(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    reading = value.strip()
    if (
        not reading
        or re.search(r"\bUNKNOWN\b", reading, flags=re.IGNORECASE)
        or re.search(r"#[0-9A-Fa-f]{6}\b", reading)
        or any(
            re.search(rf"\b{re.escape(identifier)}\b", reading, flags=re.IGNORECASE)
            for identifier, _rgb in _MARKER_COLORS
        )
    ):
        return None
    return reading


def _valid_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) for coordinate in value):
        return None
    left, top, right, bottom = (float(coordinate) for coordinate in value)
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        return None
    return left, top, right, bottom


def _same_display_region(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if left >= right or top >= bottom:
        return False
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / min(first_area, second_area) >= 0.8


__all__ = [
    "LabInstrumentReadResult",
    "LabInstrumentAgent",
    "ReadLabInstrumentsRequest",
]
