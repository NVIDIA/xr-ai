# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-selected voice updates and adaptive reasoning for one user turn."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import ChatMessage
from xr_ai_tools import Tool, ToolSet

from ._runtime import VoiceOutput

_REASONING_GUIDANCE = """<reasoning_guidance>
Reason only about unresolved choices that affect correctness. Form one compact
plan, preserve every explicit constraint, and act as soon as the required tool
and arguments are clear. Do not restate the request, revisit settled choices,
or explore alternatives that cannot change the action.
</reasoning_guidance>"""


class _PrepareWork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        description="Brief contextual acknowledgement or progress update for the user.",
    )
    use_reasoning: bool = Field(
        description="Whether subsequent model calls need deliberate hidden reasoning.",
    )


class _ProgressUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        description="Brief user-facing update containing genuinely new progress.",
    )


class _CoordinationResult(BaseModel):
    accepted: bool
    reasoning_enabled: bool


VoicePublisher = Callable[[VoiceOutput], Awaitable[None]]


_CURRENT_TURN: ContextVar[VoiceTurnController | None] = ContextVar(
    "xr_ai_voice_turn",
    default=None,
)


class VoiceTurnController:
    """Expose model-owned acknowledgement, progress, and reasoning decisions.

    The model chooses whether to call either coordination tool and supplies all
    spoken content. The controller only carries those choices safely to the
    participant's voice aggregation stream.
    """

    def __init__(
        self,
        *,
        turn_id: str,
        timestamp_us: int | None,
        publish: VoicePublisher | None,
        acknowledgement: bool,
    ) -> None:
        if not turn_id.strip():
            raise ValueError("voice turn id must not be empty")
        self.turn_id = turn_id
        self.timestamp_us = timestamp_us
        self._publish = publish
        self._acknowledgement = acknowledgement
        self._prepared = False
        self._reasoning_enabled = False
        self._tools = ToolSet(
            (
                Tool(
                    "turn__prepare_work",
                    self._prepare_description(),
                    _PrepareWork,
                    _CoordinationResult,
                    self._prepare_work,
                    examples=self._prepare_examples(),
                ),
                Tool(
                    "turn__report_progress",
                    "Use only after a meaningful new milestone when substantial work still remains. "
                    "The message must say what advanced without mentioning reasoning, agents, tools, "
                    "or implementation details. Never use as a heartbeat, repeat an acknowledgement, "
                    "or delay the final result.",
                    _ProgressUpdate,
                    _CoordinationResult,
                    self._report_progress,
                    examples=(
                        "After one independent part is complete and more requested work remains, "
                        "briefly report that completed part.",
                    ),
                ),
            )
        )

    @property
    def reasoning_enabled(self) -> bool:
        """Whether the model selected deliberate reasoning for later calls."""

        return self._reasoning_enabled

    @classmethod
    def current(cls) -> VoiceTurnController | None:
        """Return the controller active in the current asynchronous turn."""

        return _CURRENT_TURN.get()

    def child(self) -> VoiceTurnController:
        """Create an inner-agent controller sharing this turn's voice stream."""

        return VoiceTurnController(
            turn_id=self.turn_id,
            timestamp_us=self.timestamp_us,
            publish=self._publish,
            acknowledgement=False,
        )

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

        return ToolSet(dict((*tools.items(), *self._tools.items())))

    def messages(self, transcript: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
        """Add concise-reasoning guidance only after the model requests it."""

        messages = tuple(transcript)
        if not self._reasoning_enabled:
            return messages
        if messages and messages[0].role == "system" and isinstance(messages[0].content, str):
            first = ChatMessage(
                role="system",
                content=f"{messages[0].content}\n\n{_REASONING_GUIDANCE}",
            )
            return (first, *messages[1:])
        return (ChatMessage(role="system", content=_REASONING_GUIDANCE), *messages)

    def _prepare_description(self) -> str:
        if self._acknowledgement:
            delivery = (
                "The message is a short, contextual acknowledgement spoken immediately while work "
                "continues. Use natural wording specific to the request without claiming completion. "
            )
        else:
            delivery = (
                "The message is a short progress update because the outer turn may already have "
                "acknowledged the request. It must add useful context without claiming completion. "
            )
        return (
            "Call at most once, alone as the first tool call, only when the remaining work merits "
            "a user-facing update or deliberate reasoning. "
            f"{delivery}"
            "Reasoning is a high-latency exception: default use_reasoning to false. Set it true "
            "only when at least two unresolved choices interact such that their combination "
            "changes the correct or safe plan before a domain tool can be selected. Keep it false "
            "for direct tool calls, independent compound requests, ordinary routing, visual or "
            "physical-source lookup, missing-data clarification, and slow external work. Multiple "
            "steps or agents alone do not justify reasoning. Do not call at all for a direct "
            "answer or a clear operation that can finish promptly."
        )

    def _prepare_examples(self) -> tuple[str, ...]:
        if self._acknowledgement:
            return (
                "For several independent requested changes, acknowledge the overall goal in one "
                "short sentence and set use_reasoning=false.",
                "For a novel request where two unresolved constraints affect each other's safe "
                "ordering, acknowledge the goal and set use_reasoning=true.",
                "For one clear but slow external check, acknowledge the check and set "
                "use_reasoning=false.",
            )
        return (
            "For a focused instruction with multiple independent steps, set use_reasoning=false.",
            "For a focused instruction where unresolved constraints interact and no domain tool "
            "can yet be selected safely, briefly report that issue and set use_reasoning=true.",
        )

    async def _prepare_work(self, request: _PrepareWork) -> _CoordinationResult:
        if self._prepared:
            return _CoordinationResult(
                accepted=False,
                reasoning_enabled=self._reasoning_enabled,
            )
        self._prepared = True
        self._reasoning_enabled = request.use_reasoning
        logger.info(
            "voice turn prepared turn={!r} acknowledgement={} reasoning={} message={!r}",
            self.turn_id,
            self._acknowledgement,
            self._reasoning_enabled,
            request.message,
        )
        await self._speak(
            request.message,
            kind="acknowledgement" if self._acknowledgement else "progress",
        )
        return _CoordinationResult(
            accepted=True,
            reasoning_enabled=self._reasoning_enabled,
        )

    async def _report_progress(self, request: _ProgressUpdate) -> _CoordinationResult:
        logger.info(
            "voice turn progress turn={!r} message={!r}",
            self.turn_id,
            request.message,
        )
        await self._speak(request.message, kind="progress")
        return _CoordinationResult(
            accepted=True,
            reasoning_enabled=self._reasoning_enabled,
        )

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


__all__ = ["VoicePublisher", "VoiceTurnController"]
