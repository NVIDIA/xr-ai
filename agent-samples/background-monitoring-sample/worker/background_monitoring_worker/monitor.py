# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One participant-scoped background visual-monitoring task."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_hub import FrameUnavailable, ProcessorEndpoint
from xr_ai_models import VLMService
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool
from xr_ai_voice import VoiceParticipantLeft

from .events import (
    MONITOR_RECORD_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    MonitorRecord,
)

_DEFAULT_FOCUS = "important visual changes"


class _VisualDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = Field(min_length=1, max_length=2000)
    changed: bool
    summary: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_summary(self) -> _VisualDecision:
        if self.changed and not self.summary.strip():
            raise ValueError("changed observations require a summary")
        return self


class StartMonitoringRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)
    instruction: str = Field(default="", max_length=240)


class MonitoringRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)


class MonitoringState(BaseModel):
    active: bool
    instruction: str = ""
    message: str = Field(min_length=1)


def parse_monitor_response(text: str, *, baseline: bool) -> _VisualDecision:
    """Validate the VLM's bounded JSON response and normalize baseline state."""

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("monitor response did not contain a JSON object")
    data: Any = json.loads(candidate[start : end + 1])
    decision = _VisualDecision.model_validate(data)
    if baseline:
        return decision.model_copy(update={"changed": False, "summary": ""})
    if not decision.changed and decision.summary:
        return decision.model_copy(update={"summary": ""})
    return decision


class MonitorAgent(Agent):
    """Own participant-scoped visual monitoring and its lifecycle tools."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        vlm: VLMService,
        frame_max_age_s: float,
        frame_timeout_s: float,
        prompt: str,
        interval_s: float,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.images = ImageRegistry()
        self.get_current_frame = CurrentFrameTool(
            endpoint=endpoint,
            images=self.images,
            frame_max_age_s=frame_max_age_s,
            frame_timeout_s=frame_timeout_s,
        )
        self.query_image = ImageQueryTool(images=self.images, vlm=vlm)
        self.start_monitoring = Tool(
            "start_monitoring",
            "Start background visual monitoring for one participant.",
            StartMonitoringRequest,
            MonitoringState,
            self._start_monitoring,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        self.stop_monitoring = Tool(
            "stop_monitoring",
            "Stop background visual monitoring for one participant.",
            MonitoringRequest,
            MonitoringState,
            self._stop_monitoring,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        self.monitoring_status = Tool(
            "monitoring_status",
            "Report whether background visual monitoring is active for one participant.",
            MonitoringRequest,
            MonitoringState,
            self._monitoring_status,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        super().__init__(
            (
                self.get_current_frame,
                self.query_image,
                self.start_monitoring,
                self.stop_monitoring,
                self.monitoring_status,
            )
        )
        self._prompt = prompt.strip()
        self._interval_s = interval_s
        self._runtime: AgentRuntime | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._previous: dict[str, str] = {}
        self._instructions: dict[str, str] = {}
        self._stopped = False

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        """Bind the running runtime used by detached monitoring tasks."""

        if not runtime.running:
            raise RuntimeError("monitor requires a running agent runtime")
        if self._runtime is not None:
            raise RuntimeError("monitor is already started")
        self._runtime = runtime

    async def _start_monitoring(
        self,
        request: StartMonitoringRequest,
    ) -> MonitoringState:
        if self._stopped:
            raise RuntimeError("monitor is stopping")
        runtime = self._runtime
        if runtime is None or not runtime.running:
            raise RuntimeError("monitor requires a running agent runtime")
        participant_id = request.participant_id
        existing = self._tasks.get(participant_id)
        if existing is not None and not existing.done():
            instruction = self._instructions[participant_id]
            return MonitoringState(
                active=True,
                instruction=instruction,
                message=f"Background monitoring is already running. Monitoring: {instruction}.",
            )
        instruction = request.instruction.strip().rstrip(".!? ") or _DEFAULT_FOCUS
        self._previous.pop(participant_id, None)
        self._instructions[participant_id] = instruction
        task = asyncio.create_task(
            self._monitor(participant_id),
            name=f"background-monitor:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard(pid, completed)
        )
        return MonitoringState(
            active=True,
            instruction=instruction,
            message=f"Background monitoring started. Monitoring: {instruction}.",
        )

    async def _stop_monitoring(self, request: MonitoringRequest) -> MonitoringState:
        participant_id = request.participant_id
        instruction = self._instructions.get(participant_id, "")
        active = await self._cancel(participant_id)
        self._previous.pop(participant_id, None)
        self._instructions.pop(participant_id, None)
        self.get_current_frame.release(participant_id)
        if not active:
            return MonitoringState(
                active=False,
                message="Background monitoring is not running.",
            )
        return MonitoringState(
            active=False,
            instruction=instruction,
            message="Background monitoring stopped.",
        )

    async def _monitoring_status(self, request: MonitoringRequest) -> MonitoringState:
        task = self._tasks.get(request.participant_id)
        active = task is not None and not task.done()
        instruction = self._instructions.get(request.participant_id, "") if active else ""
        message = (
            f"Background monitoring is running. Monitoring: {instruction}."
            if active
            else "Background monitoring is stopped."
        )
        return MonitoringState(
            active=active,
            instruction=instruction,
            message=message,
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)
        self._previous.pop(participant_id, None)
        self._instructions.pop(participant_id, None)
        self.get_current_frame.release(participant_id)

    async def stop(self) -> None:
        """Cancel every monitoring task before the runtime closes."""

        self._stopped = True
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._previous.clear()
        self._instructions.clear()
        self.images.clear()
        self._runtime = None

    async def _monitor(self, participant_id: str) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            while True:
                record = await self._observe(participant_id)
                runtime = self._runtime
                if runtime is None:
                    return
                await runtime.publish(
                    MONITOR_RECORD_TOPIC,
                    record,
                    participant_id=participant_id,
                    source="monitor",
                )
                await asyncio.sleep(self._interval_s)

    async def _observe(self, participant_id: str) -> MonitorRecord:
        now_us = time.time_ns() // 1_000
        previous = self._previous.get(participant_id)
        instruction = self._instructions.get(participant_id, _DEFAULT_FOCUS)
        previous_text = json.dumps(previous, ensure_ascii=False) if previous else "null"
        query = (
            f"{self._prompt}\nMonitoring focus: {instruction}\n"
            f"Previous caption: {previous_text}"
        )
        try:
            frame = await self.get_current_frame.execute(
                CurrentFrameRequest(participant_id=participant_id)
            )
            result = await self.query_image.execute(
                ImageQueryRequest(image=frame.image, query=query)
            )
        except asyncio.CancelledError:
            raise
        except FrameUnavailable as exc:
            return MonitorRecord(
                timestamp_us=now_us,
                record_type="unavailable",
                error=str(exc),
            )
        except Exception as exc:
            logger.opt(exception=True).warning(
                "background monitor vision failed pid={!r}",
                participant_id,
            )
            return MonitorRecord(
                timestamp_us=now_us,
                record_type="error",
                error=str(exc),
            )
        if not result.available:
            return MonitorRecord(
                timestamp_us=now_us,
                record_type="unavailable",
                error=result.text,
            )
        try:
            decision = parse_monitor_response(result.text, baseline=previous is None)
        except (ValueError, json.JSONDecodeError) as exc:
            return MonitorRecord(
                timestamp_us=now_us,
                record_type="error",
                caption=result.text[:2000],
                error=f"invalid monitor response: {exc}",
            )
        self._previous[participant_id] = decision.caption
        return MonitorRecord(
            timestamp_us=now_us,
            record_type="baseline" if previous is None else "observation",
            caption=decision.caption,
            changed=decision.changed,
            summary=decision.summary,
        )

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
            self._previous.pop(participant_id, None)
            self._instructions.pop(participant_id, None)
            self.get_current_frame.release(participant_id)
        if task.cancelled():
            return
        if error := task.exception():
            logger.error("background monitor stopped pid={!r}: {!r}", participant_id, error)

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("monitor lifecycle requires a participant")
        return participant_id


__all__ = [
    "MonitorAgent",
    "MonitoringRequest",
    "MonitoringState",
    "StartMonitoringRequest",
    "parse_monitor_response",
]
