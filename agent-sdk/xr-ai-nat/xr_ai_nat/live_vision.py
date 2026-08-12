# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Relay-observed streaming responder for a participant's current frame."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import nemo_relay
from nemo_relay.codecs import OpenAIChatCodec
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable, LiveFrameSource, ProcessorEndpoint
from xr_ai_models import VLMService

from ._pixels import encode_image, frame_to_pil
from ._relay import headers_from_relay

_LOGGER = logging.getLogger(__name__)
_VLM_CALL_NAME = "xr-ai-vlm"
_FRAME_REDACTION = "<redacted:live-camera-frame>"


class VisionRequest(BaseModel):
    """Ask one question about a participant's current live camera frame."""

    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(description="Participant whose camera frame should be inspected.")
    query: str = Field(min_length=1, description="Question to answer from the camera frame.")


class VisionChunk(BaseModel):
    """One streamed text fragment from a current-frame answer."""

    text: str = Field(description="A partial fragment of the streamed answer text.")


class LiveVisionResponder:
    """Stream one participant-scoped answer through Relay's managed LLM path."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        vlm: VLMService,
        system_prompt: str = "",
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        if frame_max_age_s <= 0.0:
            raise ValueError("frame_max_age_s must be positive")
        if frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")
        self.endpoint = endpoint
        self.vlm = vlm
        self.system_prompt = system_prompt
        self.frames = LiveFrameSource(
            endpoint,
            max_age_s=frame_max_age_s,
            timeout_s=frame_timeout_s,
        )

    async def stream(self, request: VisionRequest) -> AsyncIterator[VisionChunk]:
        """Run one live-vision turn without exposing camera bytes to telemetry."""

        request = VisionRequest.model_validate(request)
        with nemo_relay.scope.scope(
            "look_at_current_frame",
            nemo_relay.ScopeType.Agent,
            input=request.model_dump(mode="json"),
        ) as handle:
            nemo_relay.scope_local.register_llm_sanitize_request(
                handle,
                "xr-ai-live-frame",
                0,
                _sanitize_live_frame,
            )
            async for chunk in self._stream_current(request):
                yield VisionChunk.model_validate(chunk)

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant disconnects."""

        self.frames.release(participant_id)

    async def _stream_current(self, request: VisionRequest) -> AsyncIterator[VisionChunk]:
        try:
            image_url = await self._current_image(request.participant_id)
        except FrameUnavailable as exc:
            yield VisionChunk(text=str(exc))
            return
        except Exception:
            _LOGGER.exception("Live frame conversion failed")
            yield VisionChunk(text="VLM server unavailable — please retry.")
            return

        await self.endpoint.set_status("processing", request.participant_id)
        fragments: list[str] = []
        try:
            stream = await nemo_relay.llm.stream_execute(
                _VLM_CALL_NAME,
                self._relay_request(image_url, request.query),
                self._stream_vlm,
                lambda chunk: fragments.append(_stream_text(chunk)),
                lambda: _openai_response("".join(fragments)),
                model_name=_VLM_CALL_NAME,
                codec=OpenAIChatCodec(),
                response_codec=OpenAIChatCodec(),
            )
            async for chunk in stream:
                text = _stream_text(chunk)
                if text:
                    yield VisionChunk(text=text)
        except Exception:
            _LOGGER.exception("Live VLM stream failed")
            yield VisionChunk(text="VLM server unavailable — please retry.")
        finally:
            await self.endpoint.set_status("idle", request.participant_id)

    async def _current_image(self, participant_id: str) -> str:
        frame = await self.frames.get(participant_id)
        return await asyncio.to_thread(lambda: encode_image(frame_to_pil(frame)))

    def _relay_request(self, image_url: str, query: str) -> nemo_relay.LLMRequest:
        return nemo_relay.LLMRequest(
            {},
            {
                "model": _VLM_CALL_NAME,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
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

    async def _stream_vlm(
        self,
        request: nemo_relay.LLMRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        image_url, query, system_prompt = _vision_inputs(request.content)
        async for token in self.vlm.stream(
            image_url,
            query,
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        ):
            yield {"choices": [{"delta": {"content": token}}]}


def _sanitize_live_frame(
    request: nemo_relay.LLMRequest,
    _context: nemo_relay.LlmSanitizeRequestContext,
) -> nemo_relay.LLMRequest:
    """Redact inline camera data from events without changing provider input."""

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


def _vision_inputs(content: Mapping[str, object]) -> tuple[str, str, str]:
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


def _openai_response(text: str) -> dict[str, Any]:
    return {
        "model": _VLM_CALL_NAME,
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _stream_text(raw_chunk: object) -> str:
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


__all__ = ["LiveVisionResponder", "VisionChunk", "VisionRequest"]
