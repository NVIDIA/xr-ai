# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured refusal and concise reasoning for focused render subagents."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import ToolLoopResult

from ..models import SubagentResult

_DECLINE_TOOL = "subagent__decline"

_REASONING_GUIDANCE = """<reasoning_guidance>
Reason only about unresolved interacting constraints in this focused task. Form
one compact plan and call the required domain tools as soon as their arguments
are clear. Do not restate the request, revisit settled choices, or explore
alternatives that cannot change the action.
</reasoning_guidance>"""


class _DispositionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        min_length=1,
        description="Brief user-facing result or explanation of the responsibility mismatch.",
    )


class _Disposition(_DispositionArgs):
    reroute: bool
    suggested_owner: str | None = Field(
        default=None,
        description=(
            "Responsible agent for a declined task when clear; otherwise JSON null. Never use "
            "the string 'None'."
        ),
    )


class _DeclineTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, description="Brief responsibility mismatch.")
    suggested_owner: str | None = Field(
        default=None,
        description="Responsible agent when clear; otherwise JSON null.",
    )


class _DeclinedTask(BaseModel):
    reason: str
    suggested_owner: str | None = None


async def _request_disposition(
    llm: LLMService, *, prompt: str
) -> _Disposition | None:
    try:
        response = await llm.chat(
            (
                ChatMessage(
                    role="system",
                    content=(
                        "Audit one focused agent task. Decide only whether the requested "
                        "operation is within the supplied responsibility. A task belongs here "
                        "even when it failed, lacks data, or needs clarification. Treat operations "
                        "that the responsibility says to never use this agent for as belonging "
                        "to a different agent. Judge the requested operation, not whether its "
                        "arguments can currently be resolved. Answer exactly two lines using this "
                        "format:\nDECISION: IN_SCOPE or DECISION: OUT_OF_SCOPE\nREASON: one "
                        "brief explanation"
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ),
            max_tokens=128,
            temperature=0.0,
            enable_thinking=False,
        )
    except Exception as exc:
        logger.warning("subagent disposition classification failed: {}", exc)
        return None
    logger.debug(
        "subagent disposition response content={!r} calls={!r}",
        response.content,
        response.tool_calls,
    )
    lines = [line.strip() for line in response.content.splitlines() if line.strip()]
    if not lines:
        return None
    decision = lines[0].upper()
    reason = lines[1].removeprefix("REASON:").strip() if len(lines) > 1 else ""
    if decision == "DECISION: IN_SCOPE":
        return _Disposition(reason=reason or "Accepted.", reroute=False)
    if decision == "DECISION: OUT_OF_SCOPE":
        return _Disposition(reason=reason or "Task belongs to another agent.", reroute=True)
    return None


def _declined_result(disposition: _Disposition) -> SubagentResult:
    return SubagentResult(
        result=disposition.reason,
        handled=False,
        suggested_owner=(
            None
            if disposition.suggested_owner in {None, "None", "none", "null"}
            else disposition.suggested_owner
        ),
    )


def refusal_toolset(tools: ToolSet) -> ToolSet:
    """Add an immediate structured refusal to a destructive leaf catalog."""

    decline = Tool(
        _DECLINE_TOOL,
        "Call immediately and alone when the focused instruction's requested operation is "
        "outside this agent's stated responsibility. Never call after a domain tool, for a "
        "missing target, or for a failed supported operation.",
        _DeclineTask,
        _DeclinedTask,
        lambda request: _DeclinedTask(
            reason=request.reason,
            suggested_owner=request.suggested_owner,
        ),
        return_direct=True,
        examples=(
            "Decline a request to move an existing object when this agent owns lifecycle, shape, "
            "and size but not movement.",
        ),
    )
    return ToolSet(dict((*tools.items(), (_DECLINE_TOOL, decline))))


def reasoning_messages(
    transcript: Sequence[ChatMessage], *, enabled: bool
) -> tuple[ChatMessage, ...]:
    """Add compact planning guidance when the supervisor requests reasoning."""

    messages = tuple(transcript)
    if not enabled:
        return messages
    if messages and messages[0].role == "system" and isinstance(messages[0].content, str):
        first = ChatMessage(
            role="system",
            content=f"{messages[0].content}\n\n{_REASONING_GUIDANCE}",
        )
        return (first, *messages[1:])
    return (ChatMessage(role="system", content=_REASONING_GUIDANCE), *messages)


async def adaptive_result(
    llm: LLMService,
    *,
    instruction: str,
    responsibility: str,
    result: ToolLoopResult,
    always_classify: bool = False,
    classify_scene_reads: bool = True,
) -> SubagentResult:
    """Classify a tool-free response without perturbing normal domain selection."""

    refusal = next(
        (record for record in result.tool_calls if record.call.name == _DECLINE_TOOL),
        None,
    )
    if refusal is not None:
        declined = _DeclinedTask.model_validate_json(refusal.message.content)
        return SubagentResult(
            result=declined.reason,
            handled=False,
            suggested_owner=declined.suggested_owner,
        )

    domain_calls = [
        record for record in result.tool_calls if record.call.name != "get_scene_state"
    ]
    logger.debug(
        "subagent result calls={!r}",
        [record.call.name for record in result.tool_calls],
    )
    has_owned_call = bool(domain_calls) or (
        bool(result.tool_calls) and not classify_scene_reads
    )
    if has_owned_call and not always_classify:
        return SubagentResult(result=result.content or "Done.")

    disposition = await _request_disposition(
        llm,
        prompt=(
            f"Agent responsibility: {responsibility}\n"
            f"Focused instruction: {instruction}\n"
            "Classify only the original focused instruction against the responsibility. A "
            "missing target, failed lookup, or unavailable input remains accepted when the "
            "requested operation itself is in domain. Decline only when that operation belongs "
            "outside the responsibility."
        ),
    )
    if disposition is None:
        return SubagentResult(result=result.content or "Done.")
    if not disposition.reroute:
        return SubagentResult(result=result.content or disposition.reason)
    declined = _declined_result(disposition)
    return declined.model_copy(update={"result": result.content or declined.result})


__all__ = ["adaptive_result", "reasoning_messages", "refusal_toolset"]
