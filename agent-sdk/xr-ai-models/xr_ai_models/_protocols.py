# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service protocols, message types, and capability flags.

Worker code depends on the four ``*Service`` protocols and treats every
concrete client as a structural match.  Reasoning-token field naming differs
across servers (``reasoning`` for nano_v3, ``reasoning_content`` for
nemotron_v3); ``ChatResponse.reasoning`` is the canonical post-normalization
name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Protocol, Sequence, runtime_checkable


ImageInput = bytes | Path | str
"""bytes, filesystem path, ``data:`` URL, or ``http(s)://`` URL."""

VideoInput = bytes | Path | str
"""bytes, filesystem path, ``data:`` URL, or ``http(s)://`` URL."""


@dataclass(frozen=True)
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImagePart:
    url: str
    type: Literal["image_url"] = "image_url"


@dataclass(frozen=True)
class VideoPart:
    url: str
    type: Literal["video_url"] = "video_url"


ContentPart = TextPart | ImagePart | VideoPart


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    """JSON-encoded arguments string, per the OpenAI tool-call contract."""


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart]
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    content: str
    reasoning: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Capabilities:
    streaming: bool = True
    tool_calls: bool = False
    vision: bool = False
    video: bool = False
    reasoning: bool = False


@runtime_checkable
class LLMService(Protocol):
    capabilities: Capabilities

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        timeout: float | None = None,
    ) -> ChatResponse: pass

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]: pass

    async def health(self) -> bool: pass

    async def close(self) -> None: pass


@runtime_checkable
class VLMService(Protocol):
    capabilities: Capabilities

    async def ask_image(
        self,
        image: ImageInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ChatResponse: pass

    async def ask_video(
        self,
        video: VideoInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ChatResponse: pass

    def stream(
        self,
        image: ImageInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]: pass

    async def health(self) -> bool: pass

    async def close(self) -> None: pass


@runtime_checkable
class STTService(Protocol):
    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        timeout: float | None = None,
    ) -> str: pass

    async def health(self) -> bool: pass

    async def close(self) -> None: pass


@runtime_checkable
class TTSService(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes: pass

    async def health(self) -> bool: pass

    async def close(self) -> None: pass
