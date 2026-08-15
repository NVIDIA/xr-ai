# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreground query agent with participant-scoped native tools."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, LLMService, ToolDef, VLMService
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.tool_calling import handle_tool_call, tool_definitions
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult, ImageQueryTool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)

from .events import (
    FOREGROUND_RECORD_TOPIC,
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
)
from .file_output import (
    FileOutputAgent,
    MonitoringHistoryRequest,
    MonitoringHistoryResult,
)
from .images import ParticipantImageAgent
from .monitor import (
    MonitorAgent,
    MonitoringRequest,
    MonitoringState,
    StartMonitoringRequest,
)
from .qr_instruments import (
    LabInstrumentReadResult,
    QRInstrumentAgent,
    ReadLabInstrumentsRequest,
)

CURRENT_FRAME_TOOL = "look_at_current_frame"
MONITORING_HISTORY_TOOL = "read_monitoring_history"
START_MONITORING_TOOL = "start_monitoring"
STOP_MONITORING_TOOL = "stop_monitoring"
MONITORING_STATUS_TOOL = "monitoring_status"
READ_LAB_INSTRUMENTS_TOOL = "read_lab_instruments"
START_INSTRUMENT_MONITORING_TOOL = "start_instrument_monitoring"
STOP_INSTRUMENT_MONITORING_TOOL = "stop_instrument_monitoring"
INSTRUMENT_MONITORING_STATUS_TOOL = "instrument_monitoring_status"
_MAX_TOOL_ROUNDS = 4

_CURRENT_FRAME_DESCRIPTION = (
    "Inspect the user's current camera view when the answer requires a visible "
    "fact. Do not use it for recent history or general knowledge."
)
_MONITORING_HISTORY_DESCRIPTION = "Read recent visual observations when the user asks what happened or changed."
_START_MONITORING_DESCRIPTION = (
    "Start background visual monitoring without changing the foreground. "
    "Pass the user's requested focus as instruction."
)
_STOP_MONITORING_DESCRIPTION = "Stop background visual monitoring."
_MONITORING_STATUS_DESCRIPTION = "Report whether background visual monitoring is running."
_READ_LAB_INSTRUMENTS_DESCRIPTION = (
    "Read all visible lab instrument displays and associate every meter reading with "
    "the instrument QR-code text. Always use this for instrument, gauge, meter, or "
    "display readings instead of ordinary visual inspection."
)
_START_INSTRUMENT_MONITORING_DESCRIPTION = "Continuously monitor QR-labelled lab instrument readings in the background."
_STOP_INSTRUMENT_MONITORING_DESCRIPTION = "Stop lab instrument reading monitoring."
_INSTRUMENT_MONITORING_STATUS_DESCRIPTION = "Report whether lab instrument reading monitoring is running."


class _CurrentFrameArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="A specific question about the current camera frame.",
    )


class _HistoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of newest observations to return.",
    )


class _StartMonitoringArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(
        default="",
        max_length=240,
        description="Optional concise focus for the background monitor.",
    )


class _ControlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


FOREGROUND_TOOL_DEFS = (
    ToolDef(
        CURRENT_FRAME_TOOL,
        _CURRENT_FRAME_DESCRIPTION,
        _CurrentFrameArgs.model_json_schema(),
    ),
    ToolDef(
        MONITORING_HISTORY_TOOL,
        _MONITORING_HISTORY_DESCRIPTION,
        _HistoryArgs.model_json_schema(),
    ),
    ToolDef(
        START_MONITORING_TOOL,
        _START_MONITORING_DESCRIPTION,
        _StartMonitoringArgs.model_json_schema(),
    ),
    ToolDef(
        STOP_MONITORING_TOOL,
        _STOP_MONITORING_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        MONITORING_STATUS_TOOL,
        _MONITORING_STATUS_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        READ_LAB_INSTRUMENTS_TOOL,
        _READ_LAB_INSTRUMENTS_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        START_INSTRUMENT_MONITORING_TOOL,
        _START_INSTRUMENT_MONITORING_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        STOP_INSTRUMENT_MONITORING_TOOL,
        _STOP_INSTRUMENT_MONITORING_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        INSTRUMENT_MONITORING_STATUS_TOOL,
        _INSTRUMENT_MONITORING_STATUS_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
)


class ForegroundAgent(Agent):
    """Answer accepted queries with a bounded model-selected tool loop."""

    def __init__(
        self,
        *,
        llm: LLMService,
        images: ParticipantImageAgent,
        vlm: VLMService,
        files: FileOutputAgent,
        monitor: MonitorAgent,
        qr_instruments: QRInstrumentAgent,
        prompt: str,
    ) -> None:
        self._images = images
        self.query_image = ImageQueryTool(images=images.images, vlm=vlm)
        super().__init__((self.query_image,))
        self._llm = llm
        self._files = files
        self._monitor = monitor
        self._qr_instruments = qr_instruments
        self._prompt = prompt.strip()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @subscribe(USER_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._run_turn(query, ctx),
            name=f"foreground-query:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(pid, completed))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self.stop()
        else:
            await self._cancel(participant_id)

    async def stop(self) -> None:
        """Cancel all participant turns owned by this agent."""

        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_turn(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            try:
                response, tools = await self._answer(query.text, participant_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).error("foreground query failed pid={!r}", participant_id)
                response = "I couldn't complete that request. Please try again."
                tools = []
            try:
                await ctx.publish(
                    FOREGROUND_RECORD_TOPIC,
                    ForegroundRecord(
                        timestamp_us=query.timestamp_us,
                        query=query.text,
                        response=response,
                        tools=tools,
                    ),
                )
                await ctx.publish(
                    VOICE_OUTPUT_TOPIC,
                    VoiceOutput(
                        text=response,
                        interrupt=True,
                        timestamp_us=query.timestamp_us,
                    ),
                )
            except RuntimeClosedError:
                return

    async def _answer(self, query: str, participant_id: str) -> tuple[str, list[str]]:
        tools = self._participant_tools(participant_id)
        definitions = tool_definitions(tools)
        messages = [
            ChatMessage(role="system", content=self._prompt),
            ChatMessage(role="user", content=query),
        ]
        used: list[str] = []
        for _ in range(_MAX_TOOL_ROUNDS):
            response = await self._llm.chat(
                messages,
                tools=definitions,
                max_tokens=1024,
                temperature=0.0,
                enable_thinking=False,
            )
            tool_calls = response.tool_calls or []
            if not tool_calls:
                return (response.content.strip() or "I don't have an answer."), used
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=list(tool_calls),
                )
            )
            for call in tool_calls:
                result = await handle_tool_call(call, tools)
                used.append(call.name)
                if result.return_direct:
                    return (result.message.content.strip() or "Done."), used
                messages.append(result.message)
        return "I couldn't finish that request within the tool limit.", used

    def _participant_tools(self, participant_id: str) -> ToolSet:
        async def inspect_current(request: _CurrentFrameArgs) -> ImageQueryResult:
            try:
                frame = await self._images.get_current_frame.execute(CurrentFrameRequest(participant_id=participant_id))
            except FrameUnavailable as exc:
                return ImageQueryResult(text=str(exc), available=False)
            return await self.query_image.execute(ImageQueryRequest(image=frame.image, query=request.question))

        async def read_history(request: _HistoryArgs) -> MonitoringHistoryResult:
            return await self._files.read_monitoring_history.execute(
                MonitoringHistoryRequest(
                    participant_id=participant_id,
                    limit=request.limit,
                )
            )

        async def start_monitoring(request: _StartMonitoringArgs) -> MonitoringState:
            return await self._monitor.start_monitoring.execute(
                StartMonitoringRequest(
                    participant_id=participant_id,
                    instruction=request.instruction,
                )
            )

        async def stop_monitoring(_request: _ControlArgs) -> MonitoringState:
            return await self._monitor.stop_monitoring.execute(MonitoringRequest(participant_id=participant_id))

        async def monitoring_status(_request: _ControlArgs) -> MonitoringState:
            return await self._monitor.monitoring_status.execute(MonitoringRequest(participant_id=participant_id))

        async def read_lab_instruments(
            _request: _ControlArgs,
        ) -> LabInstrumentReadResult:
            return await self._qr_instruments.read_lab_instruments.execute(
                ReadLabInstrumentsRequest(participant_id=participant_id)
            )

        async def start_instrument_monitoring(
            _request: _ControlArgs,
        ) -> MonitoringState:
            return await self._qr_instruments.start_instrument_monitoring.execute(
                MonitoringRequest(participant_id=participant_id)
            )

        async def stop_instrument_monitoring(
            _request: _ControlArgs,
        ) -> MonitoringState:
            return await self._qr_instruments.stop_instrument_monitoring.execute(
                MonitoringRequest(participant_id=participant_id)
            )

        async def instrument_monitoring_status(
            _request: _ControlArgs,
        ) -> MonitoringState:
            return await self._qr_instruments.instrument_monitoring_status.execute(
                MonitoringRequest(participant_id=participant_id)
            )

        return ToolSet(
            (
                Tool(
                    CURRENT_FRAME_TOOL,
                    _CURRENT_FRAME_DESCRIPTION,
                    _CurrentFrameArgs,
                    ImageQueryResult,
                    inspect_current,
                ),
                Tool(
                    MONITORING_HISTORY_TOOL,
                    _MONITORING_HISTORY_DESCRIPTION,
                    _HistoryArgs,
                    MonitoringHistoryResult,
                    read_history,
                ),
                Tool(
                    START_MONITORING_TOOL,
                    _START_MONITORING_DESCRIPTION,
                    _StartMonitoringArgs,
                    MonitoringState,
                    start_monitoring,
                    return_direct=True,
                    render_result=lambda result: result.message,
                ),
                Tool(
                    STOP_MONITORING_TOOL,
                    _STOP_MONITORING_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    stop_monitoring,
                    return_direct=True,
                    render_result=lambda result: result.message,
                ),
                Tool(
                    MONITORING_STATUS_TOOL,
                    _MONITORING_STATUS_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    monitoring_status,
                    return_direct=True,
                    render_result=lambda result: result.message,
                ),
                Tool(
                    READ_LAB_INSTRUMENTS_TOOL,
                    _READ_LAB_INSTRUMENTS_DESCRIPTION,
                    _ControlArgs,
                    LabInstrumentReadResult,
                    read_lab_instruments,
                    return_direct=True,
                    render_result=QRInstrumentAgent.render_readings,
                ),
                Tool(
                    START_INSTRUMENT_MONITORING_TOOL,
                    _START_INSTRUMENT_MONITORING_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    start_instrument_monitoring,
                    return_direct=True,
                    render_result=lambda result: result.message,
                ),
                Tool(
                    STOP_INSTRUMENT_MONITORING_TOOL,
                    _STOP_INSTRUMENT_MONITORING_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    stop_instrument_monitoring,
                    return_direct=True,
                    render_result=lambda result: result.message,
                ),
                Tool(
                    INSTRUMENT_MONITORING_STATUS_TOOL,
                    _INSTRUMENT_MONITORING_STATUS_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    instrument_monitoring_status,
                    return_direct=True,
                    render_result=lambda result: result.message,
                ),
            )
        )

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error("foreground query stopped pid={!r}: {!r}", participant_id, error)

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("foreground queries require a participant")
        return participant_id


__all__ = [
    "CURRENT_FRAME_TOOL",
    "FOREGROUND_TOOL_DEFS",
    "MONITORING_HISTORY_TOOL",
    "MONITORING_STATUS_TOOL",
    "READ_LAB_INSTRUMENTS_TOOL",
    "START_INSTRUMENT_MONITORING_TOOL",
    "STOP_INSTRUMENT_MONITORING_TOOL",
    "INSTRUMENT_MONITORING_STATUS_TOOL",
    "START_MONITORING_TOOL",
    "STOP_MONITORING_TOOL",
    "ForegroundAgent",
]
