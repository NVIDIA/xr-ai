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
from .instrument_monitor import InstrumentMonitorAgent
from .instruments import (
    LabInstrumentAgent,
    LabInstrumentReadResult,
    ReadLabInstrumentsRequest,
)
from .monitor import (
    MonitorAgent,
    MonitoringRequest,
    MonitoringState,
    StartMonitoringRequest,
)

CURRENT_VIEW_TOOL = "current_view"
RECENT_VISUAL_HISTORY_TOOL = "recent_visual_history"
VISUAL_MONITOR_START_TOOL = "visual_monitor__start"
VISUAL_MONITOR_STOP_TOOL = "visual_monitor__stop"
VISUAL_MONITOR_STATUS_TOOL = "visual_monitor__status"
LAB_INSTRUMENTS_READ_TOOL = "lab_instruments__read"
LAB_INSTRUMENTS_START_TOOL = "lab_instruments__start"
LAB_INSTRUMENTS_STOP_TOOL = "lab_instruments__stop"
LAB_INSTRUMENTS_STATUS_TOOL = "lab_instruments__status"
_MAX_TOOL_ROUNDS = 4

_CURRENT_VIEW_DESCRIPTION = (
    "Access the glasses camera for current or deictic visual requests, including "
    "reading unspecified visible text. The camera resolves the referent; call instead "
    "of asking for clarification. Never use for instrument readings."
)
_RECENT_VISUAL_HISTORY_DESCRIPTION = "Recent background visual observations or changes."
_VISUAL_MONITOR_START_DESCRIPTION = "Start an ordinary background visual watch for the requested focus."
_VISUAL_MONITOR_STOP_DESCRIPTION = (
    "Stop, quit, cancel, or end the ordinary background visual watch. Never use for instruments."
)
_VISUAL_MONITOR_STATUS_DESCRIPTION = "Report whether the ordinary background visual watch is running."
_LAB_INSTRUMENTS_READ_DESCRIPTION = (
    "Read current marker-labelled instruments, meters, gauges, readings, or numeric displays."
)
_LAB_INSTRUMENTS_START_DESCRIPTION = "Start continuous marker-labelled instrument reading monitoring."
_LAB_INSTRUMENTS_STOP_DESCRIPTION = "The only route to stop continuous marker-labelled instrument reading monitoring."
_LAB_INSTRUMENTS_STATUS_DESCRIPTION = (
    "Report whether continuous marker-labelled instrument reading monitoring is running."
)


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
        CURRENT_VIEW_TOOL,
        _CURRENT_VIEW_DESCRIPTION,
        _CurrentFrameArgs.model_json_schema(),
    ),
    ToolDef(
        RECENT_VISUAL_HISTORY_TOOL,
        _RECENT_VISUAL_HISTORY_DESCRIPTION,
        _HistoryArgs.model_json_schema(),
    ),
    ToolDef(
        VISUAL_MONITOR_START_TOOL,
        _VISUAL_MONITOR_START_DESCRIPTION,
        _StartMonitoringArgs.model_json_schema(),
    ),
    ToolDef(
        VISUAL_MONITOR_STOP_TOOL,
        _VISUAL_MONITOR_STOP_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        VISUAL_MONITOR_STATUS_TOOL,
        _VISUAL_MONITOR_STATUS_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        LAB_INSTRUMENTS_READ_TOOL,
        _LAB_INSTRUMENTS_READ_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        LAB_INSTRUMENTS_START_TOOL,
        _LAB_INSTRUMENTS_START_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        LAB_INSTRUMENTS_STOP_TOOL,
        _LAB_INSTRUMENTS_STOP_DESCRIPTION,
        _ControlArgs.model_json_schema(),
    ),
    ToolDef(
        LAB_INSTRUMENTS_STATUS_TOOL,
        _LAB_INSTRUMENTS_STATUS_DESCRIPTION,
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
        lab_instruments: LabInstrumentAgent,
        instrument_monitor: InstrumentMonitorAgent,
        prompt: str,
    ) -> None:
        self._images = images
        self.query_image = ImageQueryTool(images=images.images, vlm=vlm)
        super().__init__((self.query_image,))
        self._llm = llm
        self._files = files
        self._monitor = monitor
        self._lab_instruments = lab_instruments
        self._instrument_monitor = instrument_monitor
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
        for round_index in range(_MAX_TOOL_ROUNDS):
            response = await self._llm.chat(
                messages,
                tools=definitions,
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            tool_calls = response.tool_calls or []
            logger.info(
                "foreground route pid={!r} round={} tools={}",
                participant_id,
                round_index + 1,
                [call.name for call in tool_calls],
            )
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

        async def start_visual_monitor(
            request: _StartMonitoringArgs,
        ) -> MonitoringState:
            return await self._monitor.start_monitoring.execute(
                StartMonitoringRequest(
                    participant_id=participant_id,
                    instruction=request.instruction,
                )
            )

        async def stop_visual_monitor(_request: _ControlArgs) -> MonitoringState:
            return await self._monitor.stop_monitoring.execute(MonitoringRequest(participant_id=participant_id))

        async def visual_monitor_status(_request: _ControlArgs) -> MonitoringState:
            return await self._monitor.monitoring_status.execute(MonitoringRequest(participant_id=participant_id))

        async def read_lab_instruments(
            _request: _ControlArgs,
        ) -> LabInstrumentReadResult:
            return await self._lab_instruments.read_lab_instruments.execute(
                ReadLabInstrumentsRequest(participant_id=participant_id)
            )

        async def start_lab_instruments(_request: _ControlArgs) -> MonitoringState:
            return await self._instrument_monitor.start_instrument_monitoring.execute(
                MonitoringRequest(participant_id=participant_id)
            )

        async def stop_lab_instruments(_request: _ControlArgs) -> MonitoringState:
            return await self._instrument_monitor.stop_instrument_monitoring.execute(
                MonitoringRequest(participant_id=participant_id)
            )

        async def lab_instruments_status(_request: _ControlArgs) -> MonitoringState:
            return await self._instrument_monitor.instrument_monitoring_status.execute(
                MonitoringRequest(participant_id=participant_id)
            )

        def render_state(result: MonitoringState) -> str:
            return result.message

        return ToolSet(
            (
                Tool(
                    CURRENT_VIEW_TOOL,
                    _CURRENT_VIEW_DESCRIPTION,
                    _CurrentFrameArgs,
                    ImageQueryResult,
                    inspect_current,
                ),
                Tool(
                    RECENT_VISUAL_HISTORY_TOOL,
                    _RECENT_VISUAL_HISTORY_DESCRIPTION,
                    _HistoryArgs,
                    MonitoringHistoryResult,
                    read_history,
                ),
                Tool(
                    VISUAL_MONITOR_START_TOOL,
                    _VISUAL_MONITOR_START_DESCRIPTION,
                    _StartMonitoringArgs,
                    MonitoringState,
                    start_visual_monitor,
                    return_direct=True,
                    render_result=render_state,
                ),
                Tool(
                    VISUAL_MONITOR_STOP_TOOL,
                    _VISUAL_MONITOR_STOP_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    stop_visual_monitor,
                    return_direct=True,
                    render_result=render_state,
                ),
                Tool(
                    VISUAL_MONITOR_STATUS_TOOL,
                    _VISUAL_MONITOR_STATUS_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    visual_monitor_status,
                    return_direct=True,
                    render_result=render_state,
                ),
                Tool(
                    LAB_INSTRUMENTS_READ_TOOL,
                    _LAB_INSTRUMENTS_READ_DESCRIPTION,
                    _ControlArgs,
                    LabInstrumentReadResult,
                    read_lab_instruments,
                    return_direct=True,
                    render_result=LabInstrumentAgent.render_readings,
                ),
                Tool(
                    LAB_INSTRUMENTS_START_TOOL,
                    _LAB_INSTRUMENTS_START_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    start_lab_instruments,
                    return_direct=True,
                    render_result=render_state,
                ),
                Tool(
                    LAB_INSTRUMENTS_STOP_TOOL,
                    _LAB_INSTRUMENTS_STOP_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    stop_lab_instruments,
                    return_direct=True,
                    render_result=render_state,
                ),
                Tool(
                    LAB_INSTRUMENTS_STATUS_TOOL,
                    _LAB_INSTRUMENTS_STATUS_DESCRIPTION,
                    _ControlArgs,
                    MonitoringState,
                    lab_instruments_status,
                    return_direct=True,
                    render_result=render_state,
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
    "CURRENT_VIEW_TOOL",
    "FOREGROUND_TOOL_DEFS",
    "LAB_INSTRUMENTS_READ_TOOL",
    "LAB_INSTRUMENTS_START_TOOL",
    "LAB_INSTRUMENTS_STATUS_TOOL",
    "LAB_INSTRUMENTS_STOP_TOOL",
    "RECENT_VISUAL_HISTORY_TOOL",
    "VISUAL_MONITOR_START_TOOL",
    "VISUAL_MONITOR_STATUS_TOOL",
    "VISUAL_MONITOR_STOP_TOOL",
    "ForegroundAgent",
]
