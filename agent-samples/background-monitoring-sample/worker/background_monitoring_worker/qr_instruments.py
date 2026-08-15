# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QR-associated lab-instrument reading and background monitoring."""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

from loguru import logger
from PIL import Image
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
from xr_ai_tools.qr_code import DecodedQRCode, extract_qr_codes_zxing
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from .events import InstrumentReading
from .images import ParticipantImageAgent


class ReadLabInstrumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)


class LabInstrumentReadResult(BaseModel):
    readings: list[InstrumentReading] = Field(default_factory=list)
    available: bool = True
    message: str = ""


class QRInstrumentAgent(Agent):
    """Read QR-named instruments from one current frame."""

    def __init__(
        self,
        *,
        images: ParticipantImageAgent,
        vlm: VLMService,
        debug_dir: Path | None = None,
    ) -> None:
        self._images = images
        self._debug_dir = debug_dir
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        self._query_image = ImageQueryTool(images=images.images, vlm=vlm)
        self._fill_polygon = ImagePolygonFillTool(images=images.images)
        self.read_lab_instruments = Tool(
            "read_lab_instruments",
            "Read every visible lab instrument display and associate each reading with its QR-code text.",
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
            with Image.open(io.BytesIO(source)) as opened:
                codes = await asyncio.to_thread(
                extract_qr_codes_zxing,
                opened.convert("RGB"),
            )
            logger.info(
                "instrument QR scan pid={!r} image={} codes={}",
                request.participant_id,
                scan_path,
                [code.data for code in codes],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).warning("instrument frame or QR scan failed pid={!r}", request.participant_id)
            return LabInstrumentReadResult(available=False, message=str(exc))

        if not codes:
            return LabInstrumentReadResult(message="No readable QR-labelled lab instruments were found.")

        readings: list[InstrumentReading] = []
        for code in codes:
            try:
                reading = await self._read_one(frame.image, frame.timestamp_us, code)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning(
                    "instrument display read failed pid={!r} qr={!r}",
                    request.participant_id,
                    code.data,
                )
                continue
            if reading is not None:
                readings.append(reading)
        if not readings:
            return LabInstrumentReadResult(
                available=False,
                message="QR codes were found, but their instrument displays could not be read.",
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
            character if character.isalnum() or character in "-_." else "-"
            for character in participant_id
        )
        invoked_at_us = time.time_ns() // 1_000
        path = self._debug_dir / (
            f"{invoked_at_us}-{safe_participant}-"
            f"frame-{frame_timestamp_us}-seq-{sequence}.jpg"
        )
        await asyncio.to_thread(path.write_bytes, image)
        return path

    async def _read_one(
        self,
        image: ImageReference,
        timestamp_us: int,
        code: DecodedQRCode,
    ) -> InstrumentReading | None:
        if code.corners is None:
            return None
        marked = await self._fill_polygon.execute(
            ImagePolygonFillRequest(
                image=image,
                coordinates=[ImagePoint(x=point.x, y=point.y) for point in code.corners],
            )
        )
        if not marked.available or marked.image is None:
            return None
        result = await self._query_image.execute(
            ImageQueryRequest(
                image=marked.image,
                query=(
                    "The magenta polygon marks the QR code attached to the lab instrument "
                    f"named {code.data!r}. Read that instrument's nearby display. Return only "
                    "the displayed meter reading including its unit; return UNKNOWN if unreadable."
                ),
            )
        )
        reading = result.text.strip()
        if not result.available or not reading or reading.upper() == "UNKNOWN":
            return None
        return InstrumentReading(
            timestamp_us=timestamp_us,
            qr_text=code.data,
            meter_reading=reading,
        )

    @staticmethod
    def render_readings(result: LabInstrumentReadResult) -> str:
        if not result.readings:
            return result.message or "No lab instrument readings were available."
        return "; ".join(f"{reading.qr_text}: {reading.meter_reading}" for reading in result.readings)


__all__ = [
    "LabInstrumentReadResult",
    "QRInstrumentAgent",
    "ReadLabInstrumentsRequest",
]
