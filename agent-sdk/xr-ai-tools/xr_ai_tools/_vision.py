# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schemas and Relay codecs for live-vision tools."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import nemo_relay
from pydantic import BaseModel, ConfigDict, Field

VLM_CALL_NAME = "xr-ai-vlm"
_FRAME_REDACTION = "<redacted:live-camera-frame>"


class VisionRequest(BaseModel):
    """Ask one question about a participant's current live camera frame."""

    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(
        description="Participant whose camera frame should be inspected.",
    )
    query: str = Field(
        min_length=1,
        description="Question to answer from the camera frame.",
    )


class VisionResponse(BaseModel):
    """A complete answer and whether current-frame perception succeeded."""

    text: str = Field(description="Complete answer text.")
    available: bool = Field(
        default=True,
        description="Whether a current frame produced a usable visual answer.",
    )


class VisionChunk(BaseModel):
    """One text fragment from a streamed current-frame answer."""

    text: str = Field(description="A partial fragment of the streamed answer text.")


def register_frame_sanitizer() -> None:
    """Redact the current frame from events in the active tool scope."""

    nemo_relay.scope_local.register_llm_sanitize_request(
        nemo_relay.scope.get_handle(),
        "xr-ai-live-frame",
        0,
        _sanitize_live_frame,
    )


def relay_request(
    system_prompt: str,
    image_url: str,
    query: str,
) -> nemo_relay.LLMRequest:
    """Build the shared OpenAI-compatible current-frame request."""

    return nemo_relay.LLMRequest(
        {},
        {
            "model": VLM_CALL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        },
    )


def vision_inputs(content: Mapping[str, object]) -> tuple[str, str, str]:
    """Decode the image, query, and system prompt from a Relay request."""

    messages = content.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Relay VLM request must contain a message array")

    system_prompt = ""
    image_url: str | None = None
    query: str | None = None
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("Relay VLM messages must be objects")
        role = message.get("role")
        raw_content = message.get("content")
        if role == "system":
            if not isinstance(raw_content, str):
                raise TypeError("Relay VLM system content must be text")
            system_prompt = raw_content
        elif role == "user":
            candidate_image, candidate_query = _image_and_text(raw_content)
            if candidate_image is not None:
                image_url = candidate_image
            if candidate_query is not None:
                query = candidate_query

    if image_url is None or query is None:
        raise ValueError("Relay VLM request needs one image URL and one text question")
    return image_url, query, system_prompt


def openai_response(text: str) -> dict[str, Any]:
    """Build the complete Relay response used by both VLM paths."""

    return {
        "model": VLM_CALL_NAME,
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def response_text(raw_response: object) -> str:
    """Decode complete text from a Relay VLM response."""

    if not isinstance(raw_response, Mapping):
        raise TypeError("Relay VLM response must be an object")
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("Relay VLM response must contain one choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise TypeError("Relay VLM response choice must contain a message")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise TypeError("Relay VLM response content must be text")
    return content


def stream_text(raw_chunk: object) -> str:
    """Decode one text fragment from a Relay VLM stream chunk."""

    if not isinstance(raw_chunk, Mapping):
        raise TypeError("Relay VLM stream chunk must be an object")
    choices = raw_chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("Relay VLM stream chunk must contain one choice")
    delta = choices[0].get("delta")
    if not isinstance(delta, Mapping):
        raise TypeError("Relay VLM stream choice must contain a delta")
    content = delta.get("content", "")
    if not isinstance(content, str):
        raise TypeError("Relay VLM stream content must be text")
    return content


def _sanitize_live_frame(
    request: nemo_relay.LLMRequest,
    _context: nemo_relay.LlmSanitizeRequestContext,
) -> nemo_relay.LLMRequest:
    content = copy.deepcopy(request.content)
    messages = content.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            parts = message.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image = part.get("image_url")
                if isinstance(image, dict) and isinstance(image.get("url"), str):
                    image["url"] = _FRAME_REDACTION
    return nemo_relay.LLMRequest(dict(request.headers), content)


def _image_and_text(content: object) -> tuple[str | None, str | None]:
    if not isinstance(content, list):
        raise TypeError("Relay VLM user content must be a multimodal array")
    image_url: str | None = None
    query: str | None = None
    for part in content:
        if not isinstance(part, Mapping):
            raise TypeError("Relay VLM content parts must be objects")
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            query = part["text"]
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            if isinstance(image, Mapping) and isinstance(image.get("url"), str):
                image_url = image["url"]
    return image_url, query
