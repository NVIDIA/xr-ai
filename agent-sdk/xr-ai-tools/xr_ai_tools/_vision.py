# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay helpers shared by single-image, multi-image, and video-frame queries."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import count
from typing import Any

import nemo_relay

VLM_CALL_NAME = "xr-ai-vlm"
_IMAGE_REDACTION = "<redacted:image>"
_IMAGE_SANITIZER_IDS = count()
_MISSING_SCOPE_PREFIX = "not found: scope "


def _owning_scope_is_missing(error: RuntimeError, owner_uuid: str) -> bool:
    # Relay's native binding reports missing scopes as RuntimeError text.
    owner_prefix = f"{_MISSING_SCOPE_PREFIX}{owner_uuid}"
    message = str(error)
    return message == owner_prefix or message.startswith(f"{owner_prefix} ")


@contextmanager
def image_sanitizer() -> Iterator[None]:
    """Redact image locations for one VLM call in the active tool scope."""

    handle = nemo_relay.scope.get_handle()
    name = f"xr-ai-image-{next(_IMAGE_SANITIZER_IDS)}"
    nemo_relay.scope_local.register_llm_sanitize_request(
        handle,
        name,
        0,
        _sanitize_images,
    )
    try:
        yield
    finally:
        try:
            nemo_relay.scope_local.deregister_llm_sanitize_request(handle, name)
        except RuntimeError as error:
            # Popping the owner already removes all of its scope-local registrations.
            if not _owning_scope_is_missing(error, handle.uuid):
                raise


def relay_request(
    system_prompt: str,
    images: Sequence[tuple[str, int | None]],
    query: str,
) -> nemo_relay.LLMRequest:
    """Build one OpenAI-compatible request over ordered image references."""

    if not images:
        raise ValueError("at least one image is required")
    parts: list[dict[str, object]] = []
    for image_uri, timestamp_us in images:
        part: dict[str, object] = {
            "type": "image_url",
            "image_url": {"url": image_uri},
        }
        if timestamp_us is not None:
            part["timestamp_us"] = timestamp_us
        parts.append(part)
    parts.append({"type": "text", "text": query})
    return nemo_relay.LLMRequest(
        {},
        {
            "model": VLM_CALL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": parts},
            ],
        },
    )


def vision_inputs(
    content: Mapping[str, object],
) -> tuple[list[tuple[str, int | None]], str, str]:
    """Decode ordered images, optional timestamps, query, and system prompt."""

    messages = content.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Relay VLM request must contain a message array")

    system_prompt = ""
    images: list[tuple[str, int | None]] = []
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
            images, query = _images_and_text(raw_content)

    if not images or query is None:
        raise ValueError("Relay VLM request needs at least one image and one text question")
    return images, query, system_prompt


def timestamped_question(query: str, timestamps: Sequence[int | None]) -> str:
    """Attach ordered frame timing when a query represents a video timeline."""

    if not timestamps or all(timestamp is None for timestamp in timestamps):
        return query
    if any(timestamp is None for timestamp in timestamps):
        raise ValueError("video frame timestamps must be complete")
    concrete = [timestamp for timestamp in timestamps if timestamp is not None]
    origin = concrete[0]
    timeline = ", ".join(
        f"frame {index}: estimated_timestamp_us={timestamp}, "
        f"estimated_offset_s={(timestamp - origin) / 1_000_000:.6f}"
        for index, timestamp in enumerate(concrete, start=1)
    )
    return (
        "The images are video frames in chronological order. "
        f"Timeline values are approximate: {timeline}\n\n{query}"
    )


def openai_response(text: str) -> dict[str, Any]:
    """Build the complete Relay response used by all visual query paths."""

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


def visible_text(text: str) -> str:
    """Remove model-private reasoning blocks from a complete visual answer."""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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


def _sanitize_images(
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
                    image["url"] = _IMAGE_REDACTION
    return nemo_relay.LLMRequest(dict(request.headers), content)


def _images_and_text(
    content: object,
) -> tuple[list[tuple[str, int | None]], str | None]:
    if not isinstance(content, list):
        raise TypeError("Relay VLM user content must be a multimodal array")
    images: list[tuple[str, int | None]] = []
    query: str | None = None
    for part in content:
        if not isinstance(part, Mapping):
            raise TypeError("Relay VLM content parts must be objects")
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            query = part["text"]
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            if not isinstance(image, Mapping) or not isinstance(image.get("url"), str):
                continue
            timestamp = part.get("timestamp_us")
            if timestamp is not None and not isinstance(timestamp, int):
                raise TypeError("Relay VLM frame timestamp must be an integer")
            images.append((image["url"], timestamp))
    return images, query
