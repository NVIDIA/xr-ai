# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QR-associated lab-instrument reading and background monitoring."""

from __future__ import annotations

import asyncio
import io

import nemo_relay
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import VLMService
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
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
from xr_ai_voice import VoiceParticipantLeft

from .events import INSTRUMENT_READING_TOPIC, PARTICIPANT_LEFT_TOPIC, InstrumentReading
from .images import ParticipantImageAgent
from .monitor import MonitoringRequest, MonitoringState


class ReadLabInstrumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)


class LabInstrumentReadResult(BaseModel):
    readings: list[InstrumentReading] = Field(default_factory=list)
    available: bool = True
    message: str = ""


class QRInstrumentAgent(Agent):
    """Read QR-named instruments once or monitor them in the background."""

    def __init__(
        self,
        *,
        images: ParticipantImageAgent,
        vlm: VLMService,
        interval_s: float,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._images = images
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
        self.start_instrument_monitoring = Tool(
            "start_instrument_monitoring",
            "Continuously read QR-labelled lab instrument displays in the background.",
            MonitoringRequest,
            MonitoringState,
            self._start_monitoring,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        self.stop_instrument_monitoring = Tool(
            "stop_instrument_monitoring",
            "Stop background QR-labelled lab instrument monitoring.",
            MonitoringRequest,
            MonitoringState,
            self._stop_monitoring,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        self.instrument_monitoring_status = Tool(
            "instrument_monitoring_status",
            "Report whether QR-labelled lab instrument monitoring is active.",
            MonitoringRequest,
            MonitoringState,
            self._monitoring_status,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        super().__init__(
            (
                self.read_lab_instruments,
                self.start_instrument_monitoring,
                self.stop_instrument_monitoring,
                self.instrument_monitoring_status,
            )
        )
        self._interval_s = interval_s
        self._runtime: AgentRuntime | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopped = False

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if not runtime.running:
            raise RuntimeError("instrument monitor requires a running agent runtime")
        self._runtime = runtime

    async def _read_lab_instruments(
        self,
        request: ReadLabInstrumentsRequest,
    ) -> LabInstrumentReadResult:
        try:
            frame = await self._images.get_current_frame.execute(
                CurrentFrameRequest(participant_id=request.participant_id)
            )
            source = self._images.images.resolve(frame.image)
            if not isinstance(source, bytes):
                raise TypeError("current camera image must resolve to bytes")
            with Image.open(io.BytesIO(source)) as opened:
                codes = await asyncio.to_thread(extract_qr_codes_zxing, opened.convert("RGB"))
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

    async def _start_monitoring(self, request: MonitoringRequest) -> MonitoringState:
        if self._stopped:
            raise RuntimeError("instrument monitor is stopping")
        if self._runtime is None or not self._runtime.running:
            raise RuntimeError("instrument monitor requires a running agent runtime")
        task = self._tasks.get(request.participant_id)
        if task is not None and not task.done():
            return MonitoringState(
                active=True,
                message="Lab instrument monitoring is already running.",
            )
        task = asyncio.create_task(
            self._monitor(request.participant_id),
            name=f"qr-instrument-monitor:{request.participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[request.participant_id] = task
        task.add_done_callback(lambda completed, pid=request.participant_id: self._discard(pid, completed))
        return MonitoringState(active=True, message="Lab instrument monitoring started.")

    async def _stop_monitoring(self, request: MonitoringRequest) -> MonitoringState:
        active = await self._cancel(request.participant_id)
        return MonitoringState(
            active=False,
            message=("Lab instrument monitoring stopped." if active else "Lab instrument monitoring is not running."),
        )

    async def _monitoring_status(self, request: MonitoringRequest) -> MonitoringState:
        task = self._tasks.get(request.participant_id)
        active = task is not None and not task.done()
        return MonitoringState(
            active=active,
            message=("Lab instrument monitoring is running." if active else "Lab instrument monitoring is stopped."),
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            await self._cancel(participant_id)

    async def _monitor(self, participant_id: str) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            while True:
                result = await self._read_lab_instruments(ReadLabInstrumentsRequest(participant_id=participant_id))
                runtime = self._runtime
                if runtime is None:
                    return
                for reading in result.readings:
                    await runtime.publish(
                        INSTRUMENT_READING_TOPIC,
                        reading,
                        participant_id=participant_id,
                        source="qr-instrument-monitor",
                    )
                await asyncio.sleep(self._interval_s)

    async def _cancel(self, participant_id: str) -> bool:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if not task.cancelled() and (error := task.exception()):
            logger.error("instrument monitor stopped pid={!r}: {!r}", participant_id, error)

    async def stop(self) -> None:
        self._stopped = True
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runtime = None

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
