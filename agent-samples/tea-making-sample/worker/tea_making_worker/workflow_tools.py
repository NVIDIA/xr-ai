# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-bound native tools used by tea guidance."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.rag import RAGTools, RetrieveRequest, RetrieveResult
from xr_ai_tools.types import EmptyRequest, StrictRequest
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryResult,
    ImageQueryTool,
)

from .workflow_state import WorkflowSession, WorkflowStore

ChangeCallback = Callable[[], Awaitable[None]]


class CurrentViewRequest(StrictRequest):
    """A model-authored question about the current participant frame."""

    question: str = Field(min_length=1, max_length=500)


class RAGLookupRequest(StrictRequest):
    """A bounded sample-document retrieval request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=2, ge=1, le=2)


class NowResult(BaseModel):
    """Current Unix time in microseconds."""

    epoch_us: int


class TimerRequest(StrictRequest):
    """Inputs for a fresh monotonic-with-wall-clock timer reading."""

    started_at_us: int = Field(gt=0)
    duration_s: int = Field(gt=0)


class TimerResult(BaseModel):
    """Fresh elapsed, remaining, and expiry values."""

    elapsed_s: int
    remaining_s: int
    expired: bool


class TemperatureVerifyRequest(StrictRequest):
    """An exact temperature and explicit unit read from the current view."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    reading: float = Field(description="Exact observed numeric temperature.")
    unit: Literal["celsius", "fahrenheit"] = Field(
        description="Unit shown with the observed reading."
    )


class TemperatureVerifyResult(BaseModel):
    """Normalized reading compared with the active tea target."""

    reading_c: float
    target_c: float
    ready: bool


class AdvanceRequest(StrictRequest):
    """Explicit user-controlled transition request."""

    skip: bool = False


class CommitRequest(StrictRequest):
    """Atomic active-step state patch selected by an observation model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    updates: dict[str, bool | int | float | str] = Field(default_factory=dict)
    message: str = Field(default="", max_length=240)


class WorkflowControlResult(BaseModel):
    """Natural-language result of a deterministic lifecycle operation."""

    message: str


class WorkflowCommitResult(BaseModel):
    """Model-visible outcome of one deterministic commit attempt."""

    accepted: bool
    complete: bool
    message: str
    revision: int


def participant_current_view_tool(
    participant_id: str,
    current_frame: CurrentFrameTool,
    image_query: ImageQueryTool,
    *,
    timeout_s: float = 15.0,
) -> Tool[CurrentViewRequest, ImageQueryResult]:
    """Bind current-frame selection to a participant outside model arguments."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    async def inspect(request: CurrentViewRequest) -> ImageQueryResult:
        try:
            async with asyncio.timeout(timeout_s):
                frame = await current_frame.execute(
                    CurrentFrameRequest(participant_id=participant_id)
                )
                return await image_query.execute(
                    ImageQueryRequest(
                        image=frame.image,
                        query=request.question,
                    )
                )
        except FrameUnavailable as exc:
            return ImageQueryResult(
                text=f"Unable to inspect the current frame: {exc}",
                available=False,
            )
        except TimeoutError:
            return ImageQueryResult(
                text="Unable to inspect the current frame before the vision timeout.",
                available=False,
            )

    return Tool(
        "current_view",
        "Inspect this participant's current camera frame to answer a question about the current scene.",
        CurrentViewRequest,
        ImageQueryResult,
        inspect,
        render_result=lambda result: result.text,
    )


def rag_lookup_tool(
    rag: RAGTools,
) -> Tool[RAGLookupRequest, RetrieveResult]:
    """Expose bounded tea-document retrieval through the native RAG client."""

    async def retrieve(request: RAGLookupRequest) -> RetrieveResult:
        return await rag.retrieve.execute(
            RetrieveRequest(
                query=request.query,
                top_k=min(request.top_k, 2),
            )
        )

    return Tool(
        "rag_lookup",
        (
            "Retrieve tea and brewing knowledge from the sample documents. "
            "Retrieval never identifies a visible tea; exact-variety workflow "
            "values require a matching variety in the result."
        ),
        RAGLookupRequest,
        RetrieveResult,
        retrieve,
    )


def clock_now_tool() -> Tool[EmptyRequest, NowResult]:
    """Return a native deterministic wall-clock tool."""

    async def now(_request: EmptyRequest) -> NowResult:
        return NowResult(epoch_us=time.time_ns() // 1_000)

    return Tool(
        "clock__now",
        "Return current Unix time in microseconds.",
        EmptyRequest,
        NowResult,
        now,
    )


def clock_timer_tool() -> Tool[TimerRequest, TimerResult]:
    """Return a fresh native timer calculation."""

    async def timer(request: TimerRequest) -> TimerResult:
        elapsed_us = max(
            0,
            time.time_ns() // 1_000 - request.started_at_us,
        )
        duration_us = request.duration_s * 1_000_000
        return TimerResult(
            elapsed_s=elapsed_us // 1_000_000,
            remaining_s=max(
                0,
                math.ceil((duration_us - elapsed_us) / 1_000_000),
            ),
            expired=elapsed_us >= duration_us,
        )

    return Tool(
        "clock__timer",
        "Return fresh elapsed, remaining, and expiry values for a timer.",
        TimerRequest,
        TimerResult,
        timer,
    )


def temperature_verify_tool(
    session: WorkflowSession,
) -> Tool[TemperatureVerifyRequest, TemperatureVerifyResult]:
    """Bind temperature comparison to the active participant's target."""

    async def verify(
        request: TemperatureVerifyRequest,
    ) -> TemperatureVerifyResult:
        target_c = float(session.state["target_temperature_c"])
        reading_c = (
            request.reading
            if request.unit == "celsius"
            else (request.reading - 32) * 5 / 9
        )
        return TemperatureVerifyResult(
            reading_c=reading_c,
            target_c=target_c,
            ready=reading_c >= target_c,
        )

    return Tool(
        "temperature__verify",
        (
            "Compare an exact observed Celsius or Fahrenheit reading with "
            "the active tea target."
        ),
        TemperatureVerifyRequest,
        TemperatureVerifyResult,
        verify,
    )


def workflow_start_tool(
    store: WorkflowStore,
    session: WorkflowSession,
    on_change: ChangeCallback,
) -> Tool[EmptyRequest, WorkflowControlResult]:
    """Create the idle-root tool that starts tea guidance."""

    async def start(_request: EmptyRequest) -> WorkflowControlResult:
        async with session.lock:
            message = store.start(session)
        await on_change()
        return WorkflowControlResult(message=message)

    return _control_tool(
        "workflow__start",
        "Start tea guidance and capture future turns for the current step.",
        EmptyRequest,
        start,
    )


def workflow_management_tools(
    store: WorkflowStore,
    session: WorkflowSession,
    on_change: ChangeCallback,
) -> tuple[Tool[Any, WorkflowControlResult], ...]:
    """Create deterministic controls for one active participant session."""

    async def advance(request: AdvanceRequest) -> WorkflowControlResult:
        async with session.lock:
            message = store.advance(session, skip=request.skip)
        await on_change()
        return WorkflowControlResult(message=message)

    async def reset(_request: EmptyRequest) -> WorkflowControlResult:
        async with session.lock:
            message = store.reset(session)
        await on_change()
        return WorkflowControlResult(message=message)

    async def restart(_request: EmptyRequest) -> WorkflowControlResult:
        async with session.lock:
            message = store.restart(session)
        await on_change()
        return WorkflowControlResult(message=message)

    return (
        _control_tool(
            "workflow__advance",
            (
                "Move the guide forward only after a direct imperative command. "
                "Set skip true only for an explicit skip command."
            ),
            AdvanceRequest,
            advance,
        ),
        _control_tool(
            "workflow__reset",
            "Exit, stop, or reset tea guidance and return to the root assistant.",
            EmptyRequest,
            reset,
        ),
        _control_tool(
            "workflow__restart",
            "Clear progress and restart tea guidance from its first step.",
            EmptyRequest,
            restart,
        ),
        workflow_status_tool(store, session),
    )


def workflow_status_tool(
    store: WorkflowStore,
    session: WorkflowSession,
) -> Tool[EmptyRequest, WorkflowControlResult]:
    """Create a status tool valid in both idle and active foregrounds."""

    async def status(_request: EmptyRequest) -> WorkflowControlResult:
        async with session.lock:
            message = store.status(session)
        return WorkflowControlResult(message=message)

    description = (
        "Read the current tea instruction or the next step without changing guide state."
        if session.active
        else "Report whether tea guidance is idle or its current step and readiness."
    )
    return _control_tool(
        "workflow__status",
        description,
        EmptyRequest,
        status,
    )


def workflow_commit_tool(
    store: WorkflowStore,
    session: WorkflowSession,
    *,
    expected_step_id: str | None = None,
    expected_revision: int | None = None,
) -> Tool[CommitRequest, WorkflowCommitResult]:
    """Create the sole state mutation surface for observation models."""

    async def commit(request: CommitRequest) -> WorkflowCommitResult:
        async with session.lock:
            if (
                expected_step_id is not None
                and (
                    session.step_id != expected_step_id
                    or session.revision != expected_revision
                )
            ):
                return WorkflowCommitResult(
                    accepted=False,
                    complete=False,
                    message="observation is stale",
                    revision=session.revision,
                )
            result = store.commit(
                session,
                dict(request.updates),
                request.message,
            )
        return WorkflowCommitResult(
            accepted=result.accepted,
            complete=result.complete,
            message=result.message,
            revision=result.revision,
        )

    return Tool(
        "workflow__commit",
        (
            "Commit one atomic active-step state patch. Call exactly once, "
            "using empty updates and message when nothing supported changed."
        ),
        CommitRequest,
        WorkflowCommitResult,
        commit,
        return_direct=True,
    )


def named_tool_set(
    names: tuple[str, ...],
    *,
    current_view: Tool[CurrentViewRequest, ImageQueryResult],
    rag_lookup: Tool[RAGLookupRequest, RetrieveResult],
    clock_now: Tool[EmptyRequest, NowResult],
    clock_timer: Tool[TimerRequest, TimerResult],
    temperature_verify: Tool[
        TemperatureVerifyRequest,
        TemperatureVerifyResult,
    ],
) -> ToolSet:
    """Select only YAML-authorized tools from a closed native catalog."""

    catalog: dict[str, Tool[Any, Any]] = {
        tool.name: tool
        for tool in (
            current_view,
            rag_lookup,
            clock_now,
            clock_timer,
            temperature_verify,
        )
    }
    unknown = set(names) - catalog.keys()
    if unknown:
        raise ValueError(f"workflow references unknown tools: {sorted(unknown)}")
    return ToolSet({name: catalog[name] for name in names})


def _control_tool(
    name: str,
    description: str,
    request_model: type[StrictRequest],
    handler: Callable[
        [Any],
        Awaitable[WorkflowControlResult],
    ],
) -> Tool[Any, WorkflowControlResult]:
    return Tool(
        name,
        description,
        request_model,
        WorkflowControlResult,
        handler,
        return_direct=True,
        render_result=lambda result: result.message,
    )


__all__ = [
    "AdvanceRequest",
    "CommitRequest",
    "CurrentViewRequest",
    "NowResult",
    "RAGLookupRequest",
    "TemperatureVerifyRequest",
    "TemperatureVerifyResult",
    "TimerRequest",
    "TimerResult",
    "WorkflowCommitResult",
    "WorkflowControlResult",
    "clock_now_tool",
    "clock_timer_tool",
    "named_tool_set",
    "participant_current_view_tool",
    "rag_lookup_tool",
    "temperature_verify_tool",
    "workflow_commit_tool",
    "workflow_management_tools",
    "workflow_start_tool",
    "workflow_status_tool",
]
