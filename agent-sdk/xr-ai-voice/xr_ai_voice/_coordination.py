# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-selected voice updates for one user turn."""

from __future__ import annotations

from collections.abc import Awaitable
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import ChatMessage, LLMService, ToolDef
from xr_ai_tools import Tool, ToolSet

from ._runtime import VoiceOutput


class _ProgressUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        description="Brief user-facing update containing genuinely new progress.",
    )


class _CoordinationResult(BaseModel):
    accepted: bool


class _AcknowledgementDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acknowledge: bool = Field(
        description=(
            "True for external observation, application-state changes, background activity, or "
            "multi-constraint work; false for greetings and immediate informational answers."
        ),
    )
    message: str = Field(
        default="",
        max_length=120,
        description=(
            "When acknowledge is true, one natural first-person future sentence of at most "
            "twelve words (for example, beginning with I'll) naming the concrete goal without "
            "explaining a plan or claiming completion; otherwise empty."
        ),
    )


class VoicePublisher(Protocol):
    """Publish one contribution to a participant's coordinated voice turn."""

    def __call__(self, output: VoiceOutput) -> Awaitable[None]:
        """Publish ``output`` asynchronously."""

        ...


_CURRENT_TURN: ContextVar[VoiceTurnController | None] = ContextVar(
    "xr_ai_voice_turn",
    default=None,
)


class VoiceTurnController:
    """Deliver model-owned acknowledgements and progress updates for one turn.

    The controller carries separately generated acknowledgements and
    model-selected progress updates safely to the participant's voice stream.
    """

    def __init__(
        self,
        *,
        turn_id: str,
        timestamp_us: int | None,
        publish: VoicePublisher | None,
    ) -> None:
        if not turn_id.strip():
            raise ValueError("voice turn id must not be empty")
        self.turn_id = turn_id
        self.timestamp_us = timestamp_us
        self._publish = publish
        self._prepared = False
        self._tools = ToolSet(
            (
                Tool(
                    "turn__report_progress",
                    "Use only after a meaningful new milestone when substantial work still remains. "
                    "The message must say what advanced without mentioning reasoning, agents, tools, "
                    "or implementation details. Never use as a heartbeat, repeat an acknowledgement, "
                    "or delay the final result.",
                    _ProgressUpdate,
                    _CoordinationResult,
                    self._report_progress,
                ),
            )
        )

    @classmethod
    def current(cls) -> VoiceTurnController | None:
        """Return the controller active in the current asynchronous turn."""

        return _CURRENT_TURN.get()

    @contextmanager
    def activate(self) -> Iterator[VoiceTurnController]:
        """Make this controller available to nested in-process agents."""

        token: Token[VoiceTurnController | None] = _CURRENT_TURN.set(self)
        try:
            yield self
        finally:
            _CURRENT_TURN.reset(token)

    def extend(self, tools: ToolSet) -> ToolSet:
        """Return the domain tools plus this turn's private coordination tools."""

        return ToolSet(dict((*self._tools.items(), *tools.items())))

    async def acknowledge(self, message: str) -> bool:
        """Publish one model-written acknowledgement before delegated work starts."""

        if self._prepared:
            return False
        text = message.strip()
        if not text:
            raise ValueError("voice acknowledgement must not be empty")
        self._prepared = True
        await self._speak(text, kind="acknowledgement")
        return True

    async def _report_progress(self, request: _ProgressUpdate) -> _CoordinationResult:
        logger.info(
            "voice turn progress turn={!r} message={!r}",
            self.turn_id,
            request.message,
        )
        await self._speak(request.message, kind="progress")
        return _CoordinationResult(accepted=True)

    async def _speak(self, text: str, *, kind: str) -> None:
        if self._publish is None:
            return
        await self._publish(
            VoiceOutput(
                text=text.strip(),
                timestamp_us=self.timestamp_us,
                kind=kind,
                turn_id=self.turn_id,
            )
        )


async def _acknowledge_if_needed(
    controller: VoiceTurnController,
    llm: LLMService,
    request: str,
    *,
    context: str,
) -> None:
    """Generate a low-latency acknowledgement independently of domain work."""

    try:
        response = await llm.chat(
            (
                ChatMessage(
                    role="system",
                    content=(
                        f"Classify this {context} request by calling the decision tool exactly "
                        "once. Set acknowledge=true when work continues after this decision: "
                        "external observation, an application-state change, background activity, "
                        "or reconciling multiple constraints. Set it false only for an immediate "
                        "conversational or informational answer. Do not answer the request itself."
                        " A greeting or capability question is false. Inspecting surroundings, "
                        "changing application state, or starting a watch is true."
                    ),
                ),
                ChatMessage(role="user", content=request),
            ),
            tools=(
                ToolDef(
                    name="turn__acknowledgement_decision",
                    description=(
                        "Return whether to acknowledge work that will continue. Greetings and "
                        "capability questions are false; observation, state changes, background "
                        "activity, and multi-constraint tasks are true."
                    ),
                    parameters=_AcknowledgementDecision.model_json_schema(),
                ),
            ),
            max_tokens=64,
            temperature=0.0,
            enable_thinking=False,
        )
    except Exception as error:
        logger.warning("voice acknowledgement generation failed: {}", error)
        return
    calls = tuple(response.tool_calls or ())
    if len(calls) != 1 or calls[0].name != "turn__acknowledgement_decision":
        return
    try:
        decision = _AcknowledgementDecision.model_validate_json(calls[0].arguments)
    except ValueError:
        return
    if decision.acknowledge and decision.message.strip():
        await controller.acknowledge(decision.message)


__all__ = ["VoicePublisher", "VoiceTurnController"]
