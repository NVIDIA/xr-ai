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
        palette = dict(_MARKER_COLORS)
        legend = ", ".join(
            f"{color_name}=#{red:02X}{green:02X}{blue:02X}"
            for color_name in color_keys
            for red, green, blue in (palette[color_name],)
        )
        return (
            "Read all color-block-marked instruments together. The color-name identifiers and "
            f"their exact RGB values are {legend}. The requested identifiers are: {keys}. Return "
            "one JSON object with exactly those lowercase color names as keys. Each value must be "
            "the reading and unit from that color block's own physical instrument, or UNKNOWN. "
            "Never assign one display to multiple color blocks."
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
    duplicate_key = False

    def normalized_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate_key
        normalized: dict[str, object] = {}
        for key, value in pairs:
            normalized_key = key.strip().lower()
            if normalized_key in normalized:
                duplicate_key = True
            normalized[normalized_key] = value
        return normalized

    try:
        payload = json.loads(
            visible[start : end + 1],
            object_pairs_hook=normalized_object,
        )
    except json.JSONDecodeError:
        return None
    expected_keys = {color_name: color_name.strip().lower() for color_name in color_keys}
    if duplicate_key or len(set(expected_keys.values())) != len(color_keys):
        return None
    if not isinstance(payload, dict) or set(payload) != set(expected_keys.values()):
        return None
    if not all(isinstance(value, str) and value.strip() for value in payload.values()):
        return None
    return {
        color_name: payload[normalized_name].strip()
        for color_name, normalized_name in expected_keys.items()
    }


__all__ = [
    "LabInstrumentReadResult",
    "LabInstrumentAgent",
    "ReadLabInstrumentsRequest",
]
