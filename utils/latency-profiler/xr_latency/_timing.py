# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timed wrappers around the typed model services."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from xr_ai_models import ChatMessage, LLMService


def now_ms() -> float:
    """Monotonic clock in milliseconds."""
    return time.perf_counter() * 1000.0


@dataclass
class StreamTiming:
    """Timestamps (ms since request start) for one streamed chat completion."""

    first_event: float | None = None
    first_reasoning: float | None = None
    first_text: float | None = None
    last_event: float | None = None
    total: float = 0.0
    text_chunks: int = 0
    reasoning_chunks: int = 0
    text_chars: int = 0
    reasoning_chars: int = 0
    tool_calls: int = 0
    finish_reason: str | None = None
    text_chunk_times: list[float] = field(default_factory=list)

    def as_metrics(self) -> dict[str, Any]:
        """Flat metrics for storage (the per-chunk timeline is kept only when small)."""
        out = {k: v for k, v in self.__dict__.items() if k != "text_chunk_times"}
        if len(self.text_chunk_times) <= 600:
            out["text_chunk_times"] = [round(t, 2) for t in self.text_chunk_times]
        return out


async def timed_stream(llm: LLMService, messages: Sequence[ChatMessage], **kwargs: Any) -> StreamTiming:
    """Run ``stream_events`` to completion, recording when each kind of event first arrives."""
    timing = StreamTiming()
    start = now_ms()
    async for event in llm.stream_events(messages, **kwargs):
        t = now_ms() - start
        if timing.first_event is None:
            timing.first_event = t
        timing.last_event = t
        if event.kind == "reasoning":
            if timing.first_reasoning is None:
                timing.first_reasoning = t
            timing.reasoning_chunks += 1
            timing.reasoning_chars += len(event.reasoning or "")
        elif event.kind == "text":
            if timing.first_text is None:
                timing.first_text = t
            timing.text_chunks += 1
            timing.text_chars += len(event.text or "")
            timing.text_chunk_times.append(t)
        elif event.kind == "tool_call":
            timing.tool_calls += 1
        elif event.kind == "finish":
            timing.finish_reason = event.finish_reason
    timing.total = now_ms() - start
    return timing


@dataclass
class ChatTiming:
    """Wall time and server-reported token usage for one non-streaming completion."""

    total: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_chars: int
    text_chars: int
    tool_calls: int
    finish_reason: str | None

    def as_metrics(self) -> dict[str, Any]:
        """Flat metrics for storage."""
        return dict(self.__dict__)


async def timed_chat(llm: LLMService, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatTiming:
    """Run one non-streaming ``chat`` and return wall time plus usage."""
    start = now_ms()
    response = await llm.chat(messages, **kwargs)
    total = now_ms() - start
    usage = response.raw.get("usage") or {}
    return ChatTiming(
        total=total,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_chars=len(response.reasoning or ""),
        text_chars=len(response.content or ""),
        tool_calls=len(response.tool_calls or ()),
        finish_reason=response.finish_reason,
    )
