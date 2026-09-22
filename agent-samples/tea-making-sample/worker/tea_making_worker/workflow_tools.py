# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-bound native tools used by tea guidance."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
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
_NAMED_TOOL_NAMES = frozenset(
    {
        "current_view",
        "rag_lookup",
        "clock__now",
        "clock__timer",
        "temperature__threshold",
        "temperature__verify",
    }
)


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


class TemperatureThresholdRequest(StrictRequest):
    """An exact temperature to compare with a supplied Celsius threshold."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    reading: float = Field(description="Exact observed numeric temperature.")
    unit: Literal["celsius", "fahrenheit"] = Field(
        description="Unit shown with the observed reading."
    )
    threshold_c: float = Field(description="Celsius threshold for comparison.")


class TemperatureThresholdResult(BaseModel):
    """Normalized reading and deterministic strict threshold result."""

    reading_c: float
    threshold_c: float
    above: bool


class AdvanceRequest(StrictRequest):
    """Explicit user-controlled transition request."""

    skip: bool = False


class CommitRequest(StrictRequest):
    """Atomic active-step state patch selected by an observation model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    updates: dict[str, bool | int | float | str] = Field(
        default_factory=dict,
        description=(
            "Sparse patch containing only positively supported new values. Use an empty object "
            "when a writable fact is false, absent, unclear, contradicted, or otherwise unsupported; "
            "never write false, zero, null, unknown, or placeholder values for those facts."
        ),
    )
    message: str = Field(
        default="",
        max_length=240,
        description="Short user message only for a real non-completing state change; otherwise empty.",
    )
    evidence: Literal["rejected", "unknown"] = Field(
        default="rejected",
        description=(
            "Outcome for a non-completing observation. Use unknown only when "
            "the required fact cannot be determined; completion implies acceptance."
        ),
    )


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
        examples=(
            "'Is the kettle boiling right now?' requires current_view when it is available.",
            "'What should I do at this step?' is procedural and does not inspect the camera.",
        ),
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
            "USE WHEN: a user requests any tea, brewing, or hot-water fact, or an identification "
            "workflow has a visible exact tea name but lacks a required temperature or duration. "
            "Always retrieve instead of using model memory. DO NOT USE WHEN: the exact tea name is "
            "not visibly identified; the observation already supplies both a unit-bearing brewing "
            "temperature and a unit-bearing duration; general knowledge, calculation, live visual "
            "evidence, or starting/managing a guide. "
            "Retrieval never identifies a visible tea; exact-variety workflow "
            "values require a matching variety in the result."
        ),
        RAGLookupRequest,
        RetrieveResult,
        retrieve,
        examples=(
            "'How hot should I brew a dark oolong?' uses rag_lookup.",
            "'Does fermented tea contain caffeine?' uses rag_lookup rather than model memory.",
            "A visible named tea with missing package brewing values uses rag_lookup before commit.",
            "A package with a complete unit-bearing temperature and duration commits without retrieval.",
            "'What is the capital of Peru?' is general knowledge and does not use rag_lookup.",
        ),
    )


def clock_now_tool() -> Tool[EmptyRequest, NowResult]:
    """Return a native deterministic wall-clock tool."""

    async def now(_request: EmptyRequest) -> NowResult:
        return NowResult(epoch_us=time.time_ns() // 1_000)

    return Tool(
        "clock__now",
        "USE WHEN: a workflow contract requires a fresh timestamp after its physical start "
        "condition is positively observed. When visible liquid and visible tea-liquid contact "
        "establish the start of steeping, call this before the state commit. DO NOT USE WHEN: "
        "contact is absent or unclear. Never invent, copy from visible text, or substitute a timestamp.",
        EmptyRequest,
        NowResult,
        now,
        examples=(
            "Visible tea-water contact that starts a timer requires clock__now before a state commit.",
            "A timestamp printed beside visible contact is data, not clock evidence; call clock__now.",
            "Unclear contact or a dry tea bag does not require a timestamp.",
        ),
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
        "Return fresh elapsed, remaining, and expiry values when the user asks about timer time, "
        "completion, or readiness. Do not substitute workflow status for a timer question.",
        TimerRequest,
        TimerResult,
        timer,
        examples=(
            "'How much longer until the tea is ready?' uses clock__timer.",
            "'Which guide step are we on?' is workflow state, not a timer request.",
        ),
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


def temperature_threshold_tool() -> Tool[
    TemperatureThresholdRequest,
    TemperatureThresholdResult,
]:
    """Compare a unit-bearing temperature with an explicit threshold."""

    async def compare(
        request: TemperatureThresholdRequest,
    ) -> TemperatureThresholdResult:
        reading_c = (
            request.reading
            if request.unit == "celsius"
            else (request.reading - 32) * 5 / 9
        )
        return TemperatureThresholdResult(
            reading_c=reading_c,
            threshold_c=request.threshold_c,
            above=reading_c > request.threshold_c,
        )

    return Tool(
        "temperature__threshold",
        (
            "Determine whether an exact current Celsius or Fahrenheit reading is strictly above "
            "an explicit Celsius threshold. The reading must describe the present measured "
            "temperature, not a printed recipe target or instruction."
        ),
        TemperatureThresholdRequest,
        TemperatureThresholdResult,
        compare,
        examples=(
            "A live display reading may be compared with the step threshold.",
            "A label saying 'heat to 90 degrees' is a target, not a current reading.",
        ),
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
        "USE WHEN: the user directly requests starting step-by-step tea guidance now. DO NOT USE "
        "WHEN: tea facts, capability/how-to questions, hypothetical starts, quotations, reports, "
        "or negations.",
        EmptyRequest,
        start,
        examples=(
            "'Please walk me through making tea' starts the guide.",
            "'If I wanted a guide, could you start one?' is hypothetical and does not start it.",
        ),
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
                "USE WHEN: the user directly commands the active tea guide to move forward now; "
                "always call even when supplied state looks incomplete because the tool alone decides "
                "readiness. Set skip=false for continue/next/move-on/proceed commands and "
                "skip=true only for bypass/skip-this-step commands. DO NOT USE WHEN: a question, "
                "quotation, hypothetical, deliberation, negation, report, or unrelated wording."
            ),
            AdvanceRequest,
            advance,
            examples=(
                "'Please continue to the next step' uses skip=false.",
                "'Go ahead with the guide' uses skip=false even if the current step may be incomplete.",
                "'Could you skip this step?' uses skip=true.",
                "'Should I continue after the water boils?' is a question and does not advance.",
                "'And after that?' asks for guidance and does not advance.",
            ),
        ),
        _control_tool(
            "workflow__reset",
            "USE WHEN: the user directly commands exiting, stopping, resetting, cancelling, or "
            "clearing the active tea guide now. DO NOT USE WHEN: restart/start-over, a question, "
            "quotation, hypothetical, reported speech, negation, word discussion, or unrelated "
            "reset/cancel wording.",
            EmptyRequest,
            reset,
            examples=(
                "'Please stop the tea guide' resets it.",
                "'How do I stop the guide?' is informational and does not reset it.",
                "'The recipe says to stop the guide' is reported speech and does not reset it.",
                "'The screen literally says “cancel the walkthrough”' quotes text and does not reset it.",
            ),
        ),
        _control_tool(
            "workflow__restart",
            "USE WHEN: the user directly commands restarting the active tea guide from step one "
            "now; this owns restart/start-over/begin-again requests instead of reset. DO NOT USE "
            "WHEN: a question, quotation, hypothetical, report, or negation.",
            EmptyRequest,
            restart,
            examples=(
                "'Start the tea guide over' restarts it rather than resetting it.",
                "'Begin my walkthrough again' restarts it rather than resetting it.",
                "'Would restarting erase progress?' is a question and does not restart it.",
            ),
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

    return _control_tool(
        "workflow__status",
        "USE WHEN: the user explicitly asks for the active guide's actual status, step, or "
        "progress; always call even when supplied context appears to contain the answer. DO NOT "
        "USE WHEN: timer/readiness, procedural or what-to-do questions, a "
        "command, negation, another unavailable capability, or a live fact about an object, "
        "temperature, or visible scene. When a requested evidence capability is absent, call no "
        "substitute.",
        EmptyRequest,
        status,
        examples=(
            "'Which step are we on?' uses workflow__status.",
            "'How long until steeping finishes?' uses the timer capability, not workflow__status.",
            "A request for guide details, instructions, or an overview is procedural and never "
            "uses workflow__status.",
            "A request for a live appliance reading never uses workflow__status; without a visual "
            "capability, answer that the live reading is unavailable.",
        ),
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
            "Finish one observation by committing an atomic sparse active-step patch. Call exactly "
            "once after all required evidence tools. Include only positively supported new values; "
            "use empty updates and message when nothing supported changed. Never encode an "
            "unsupported fact as false, zero, null, or a placeholder."
        ),
        CommitRequest,
        WorkflowCommitResult,
        commit,
        return_direct=True,
        examples=(
            "Unclear or missing evidence finishes with updates={} rather than a false or null value.",
            "Do not commit a partial observation while the step requires an available evidence tool.",
            "After a required clock, retrieval, or comparison result, commit its supported state once.",
        ),
    )


def named_tool_set(
    names: tuple[str, ...],
    *,
    current_view: Tool[CurrentViewRequest, ImageQueryResult],
    rag_lookup: Tool[RAGLookupRequest, RetrieveResult],
    clock_now: Tool[EmptyRequest, NowResult],
    clock_timer: Tool[TimerRequest, TimerResult],
    temperature_threshold: Tool[
        TemperatureThresholdRequest,
        TemperatureThresholdResult,
    ],
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
            temperature_threshold,
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
    *,
    examples: Sequence[str] = (),
) -> Tool[Any, WorkflowControlResult]:
    return Tool(
        name,
        description,
        request_model,
        WorkflowControlResult,
        handler,
        return_direct=True,
        render_result=lambda result: result.message,
        examples=examples,
    )


__all__ = [
    "AdvanceRequest",
    "CommitRequest",
    "CurrentViewRequest",
    "NowResult",
    "RAGLookupRequest",
    "TemperatureThresholdRequest",
    "TemperatureThresholdResult",
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
    "temperature_threshold_tool",
    "temperature_verify_tool",
    "workflow_commit_tool",
    "workflow_management_tools",
    "workflow_start_tool",
    "workflow_status_tool",
]
