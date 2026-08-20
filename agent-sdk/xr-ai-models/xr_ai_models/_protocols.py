# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service protocols, message types, and capability flags.

Worker code depends on the ``*Service`` protocols and treats every
concrete client as a structural match.  Reasoning-token field naming differs
across servers (``reasoning`` for nano_v3, ``reasoning_content`` for
nemotron_v3); ``ChatResponse.reasoning`` is the canonical post-normalization
name.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Mapping, Protocol, Sequence, runtime_checkable


ImageInput = bytes | Path | str
"""Image bytes, a filesystem path, a ``data:`` URL, or an HTTP(S) URL."""

VideoInput = bytes | Path | str
"""Video bytes, a filesystem path, a ``data:`` URL, or an HTTP(S) URL."""


@dataclass(frozen=True)
class TextPart:
    """Text content in a multimodal chat message."""

    text: str
    """The text presented to the model."""

    type: Literal["text"] = "text"
    """The OpenAI-compatible discriminator for this content part."""


@dataclass(frozen=True)
class ImagePart:
    """An image reference in a multimodal chat message."""

    url: str
    """An HTTP(S) or ``data:`` URL containing the image."""

    type: Literal["image_url"] = "image_url"
    """The OpenAI-compatible discriminator for this content part."""


@dataclass(frozen=True)
class VideoPart:
    """A video reference in a multimodal chat message."""

    url: str
    """An HTTP(S) or ``data:`` URL containing the video."""

    type: Literal["video_url"] = "video_url"
    """The OpenAI-compatible discriminator for this content part."""


ContentPart = TextPart | ImagePart | VideoPart
"""A supported text, image, or video part in a chat message."""


@dataclass(frozen=True)
class ToolCall:
    """A function-tool invocation requested by a model."""

    id: str
    """The provider-assigned identifier used to submit the tool result."""

    name: str
    """The function name requested by the model."""

    arguments: str
    """JSON-encoded arguments string, per the OpenAI tool-call contract."""


@dataclass(frozen=True)
class ToolDef:
    """A function tool definition exposed to a chat model."""

    name: str
    """The function name the model uses in a :class:`ToolCall`."""

    description: str
    """A natural-language description of what the function does."""

    parameters: dict[str, Any]
    """The function parameters as a JSON Schema object."""

    def to_openai(self) -> dict[str, Any]:
        """Return this definition in OpenAI's function-tool wire format."""

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
    """One input turn in an LLM or VLM conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    """The participant role associated with the message."""

    content: str | list[ContentPart]
    """Plain text or ordered multimodal content supplied by the participant."""

    tool_calls: list[ToolCall] | None = None
    """Tool calls emitted by an assistant turn, when present."""

    tool_call_id: str | None = None
    """The call identifier answered by a tool-result message, when applicable."""


@dataclass(frozen=True)
class ChatResponse:
    """A normalized non-streaming response from a chat model."""

    content: str
    """The assistant's user-visible response text."""

    reasoning: str | None
    """Normalized model reasoning, when the endpoint returns it."""

    tool_calls: list[ToolCall] | None
    """Function-tool invocations requested by the model, when present."""

    finish_reason: str | None
    """The provider's reason for ending generation, when supplied."""

    raw: dict[str, Any]
    """The unmodified provider response object."""


@dataclass(frozen=True)
class Capabilities:
    """Features supported by a configured model endpoint."""

    streaming: bool = True
    """Whether the endpoint supports token streaming."""

    tool_calls: bool = False
    """Whether the endpoint supports function-tool calls."""

    vision: bool = False
    """Whether the endpoint accepts image inputs."""

    video: bool = False
    """Whether the endpoint accepts video inputs."""

    reasoning: bool = False
    """Whether the endpoint can return model reasoning."""


OCRMergeLevel = Literal["word", "sentence", "paragraph"]
"""Granularity used to merge neighboring OCR detections."""


@dataclass(frozen=True)
class OCRCapabilities:
    """Structured output features supported by an OCR service."""

    merge_levels: tuple[OCRMergeLevel, ...]
    """Detection granularities accepted by the service."""

    structured_detections: bool
    """Whether the service returns separately addressable text regions."""

    bounding_boxes: bool
    """Whether structured regions include normalized geometry."""

    confidence_scores: bool
    """Whether structured regions include confidence scores."""

    reading_order: bool
    """Whether structured regions are ordered for sequential reading."""


@dataclass(frozen=True)
class OCRPoint:
    """One normalized image coordinate returned by an OCR backend."""

    x: float
    """Horizontal coordinate normalized to the image width."""

    y: float
    """Vertical coordinate normalized to the image height."""

    def __post_init__(self) -> None:
        """Reject non-finite coordinates outside the normalized image."""

        if not math.isfinite(self.x) or not 0.0 <= self.x <= 1.0:
            raise ValueError("OCR point x must be finite and between 0 and 1")
        if not math.isfinite(self.y) or not 0.0 <= self.y <= 1.0:
            raise ValueError("OCR point y must be finite and between 0 and 1")


@dataclass(frozen=True)
class OCRDetection:
    """One recognized text region in reading order."""

    text: str
    """Text recognized in this region."""

    confidence: float | None = None
    """Backend confidence score, when supplied."""

    bounding_box: tuple[OCRPoint, ...] = ()
    """Ordered normalized vertices surrounding the recognized region."""

    def __post_init__(self) -> None:
        """Reject confidence scores outside the normalized range."""

        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("OCR confidence must be finite and between 0 and 1")


@dataclass(frozen=True)
class OCRResponse:
    """Backend-neutral OCR output for one image."""

    text: str
    """All recognized text in reading order."""

    detections: tuple[OCRDetection, ...]
    """Structured text regions in reading order."""

    model: str | None
    """Backend model identifier, when supplied."""

    raw: dict[str, Any]
    """Unmodified provider response object."""


@runtime_checkable
class LLMService(Protocol):
    """Structural interface for text chat-completion services."""

    capabilities: Capabilities
    """Features supported by this service."""

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
        headers: Mapping[str, str] | None = None,
    ) -> ChatResponse:
        """Generate one complete response for a sequence of chat messages."""

        pass

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
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream user-visible response text for a sequence of chat messages."""

        pass

    async def health(self) -> bool:
        """Return whether the configured endpoint is ready for requests."""

        pass

    async def close(self) -> None:
        """Release resources owned by the service."""

        pass


@runtime_checkable
class VLMService(Protocol):
    """Structural interface for visual chat-completion services."""

    capabilities: Capabilities
    """Features supported by this service."""

    async def ask_image(
        self,
        image: ImageInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ChatResponse:
        """Generate one response to a question about an image."""

        pass

    async def ask_images(
        self,
        images: Sequence[ImageInput],
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ChatResponse:
        """Generate one response to a question about multiple images."""

        pass

    async def ask_video(
        self,
        video: VideoInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ChatResponse:
        """Generate one response to a question about a video."""

        pass

    def stream(
        self,
        image: ImageInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response text for a question about an image."""

        pass

    def stream_images(
        self,
        images: Sequence[ImageInput],
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response text for a question about multiple images."""

        pass

    async def health(self) -> bool:
        """Return whether the configured endpoint is ready for requests."""

        pass

    async def close(self) -> None:
        """Release resources owned by the service."""

        pass


@runtime_checkable
class OCRService(Protocol):
    """Structural interface for optical character recognition services."""

    capabilities: OCRCapabilities
    """Structured output features supported by this service."""

    async def recognize(
        self,
        image: ImageInput,
        *,
        merge_level: OCRMergeLevel = "paragraph",
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> OCRResponse:
        """Recognize visible text in one image."""

        pass

    async def health(self) -> bool:
        """Return whether the configured endpoint is ready for requests."""

        pass

    async def close(self) -> None:
        """Release resources owned by the service."""

        pass


@runtime_checkable
class STTService(Protocol):
    """Structural interface for speech-to-text services."""

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        timeout: float | None = None,
    ) -> str:
        """Transcribe WAV data or 16-bit PCM audio into text."""

        pass

    async def health(self) -> bool:
        """Return whether the configured endpoint is ready for requests."""

        pass

    async def close(self) -> None:
        """Release resources owned by the service."""

        pass


@runtime_checkable
class TTSService(Protocol):
    """Structural interface for text-to-speech services."""

    async def synthesize(
        self,
        text: str,
        *,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes:
        """Synthesize text and return audio in the requested format."""

        pass

    async def health(self) -> bool:
        """Return whether the configured endpoint is ready for requests."""

        pass

    async def close(self) -> None:
        """Release resources owned by the service."""

        pass


@runtime_checkable
class EmbeddingService(Protocol):
    """Structural interface for text-embedding services."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> list[list[float]]:
        """Embed text strings, returning one vector for each input in order."""

        pass

    async def health(self) -> bool:
        """Return whether the configured endpoint is ready for requests."""

        pass

    async def close(self) -> None:
        """Release resources owned by the service."""

        pass
