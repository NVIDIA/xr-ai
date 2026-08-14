# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite and streaming VLM tools over caller-selected images."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import nemo_relay
from nemo_relay.codecs import OpenAIChatCodec
from pydantic import BaseModel, Field
from xr_ai_models import VLMService

from ._relay import headers_from_relay
from ._vision import (
    VLM_CALL_NAME,
    openai_response,
    register_image_sanitizer,
    relay_request,
    response_text,
    stream_text,
    visible_text,
    vision_inputs,
)
from .async_tools import AsyncTool
from .image import ImageReference, ImageRegistry
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)


class ImageQueryRequest(StrictRequest):
    image: ImageReference = Field(description="Image selected by another tool or caller.")
    query: str = Field(min_length=1, description="Question to answer from the image.")


class ImageQueryResult(BaseModel):
    text: str = Field(description="Complete answer text.")
    available: bool = Field(
        default=True,
        description="Whether the image produced a usable visual answer.",
    )


class ImageQueryChunk(BaseModel):
    text: str = Field(description="A partial fragment of the streamed answer text.")


class ImageQueryTool(Tool[ImageQueryRequest, ImageQueryResult]):
    """Answer a question about any caller-selected image."""

    def __init__(
        self,
        *,
        images: ImageRegistry,
        vlm: VLMService,
        system_prompt: str = "",
    ) -> None:
        self.images = images
        self.vlm = vlm
        self.system_prompt = system_prompt
        super().__init__(
            "query_image",
            "Answer a question about an image reference returned by an image tool or supplied by the caller.",
            ImageQueryRequest,
            ImageQueryResult,
            self._answer,
            render_result=lambda result: result.text,
        )

    async def _answer(self, request: ImageQueryRequest) -> ImageQueryResult:
        try:
            register_image_sanitizer()
            response = await nemo_relay.llm.execute(
                VLM_CALL_NAME,
                relay_request(self.system_prompt, request.image.uri, request.query),
                self._ask_vlm,
                model_name=VLM_CALL_NAME,
                codec=OpenAIChatCodec(),
                response_codec=OpenAIChatCodec(),
            )
            text = visible_text(response_text(response))
            if not text:
                return ImageQueryResult(
                    text="The image did not produce an answer.",
                    available=False,
                )
            return ImageQueryResult(text=text)
        except Exception:
            _LOGGER.exception("Image VLM request failed")
            return ImageQueryResult(
                text="VLM server unavailable — please retry.",
                available=False,
            )

    async def _ask_vlm(self, request: nemo_relay.LLMRequest) -> dict[str, Any]:
        image_uri, query, system_prompt = vision_inputs(request.content)
        text = await self.vlm.ask_image(
            self.images.resolve(ImageReference(uri=image_uri)),
            query,
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        )
        return openai_response(text.content)


class StreamingImageQueryTool(AsyncTool[ImageQueryRequest, ImageQueryChunk]):
    """Stream an answer about any caller-selected image."""

    def __init__(
        self,
        *,
        images: ImageRegistry,
        vlm: VLMService,
        system_prompt: str = "",
    ) -> None:
        self.images = images
        self.vlm = vlm
        self.system_prompt = system_prompt
        super().__init__(
            "stream_image_query",
            "Stream an answer about an image reference returned by an image tool or supplied by the caller.",
            ImageQueryRequest,
            ImageQueryChunk,
            self._stream,
        )

    async def _stream(
        self,
        request: ImageQueryRequest,
    ) -> AsyncIterator[ImageQueryChunk]:
        fragments: list[str] = []
        emitted_output = False
        try:
            register_image_sanitizer()
            stream = await nemo_relay.llm.stream_execute(
                VLM_CALL_NAME,
                relay_request(self.system_prompt, request.image.uri, request.query),
                self._stream_vlm,
                lambda chunk: fragments.append(stream_text(chunk)),
                lambda: openai_response("".join(fragments)),
                model_name=VLM_CALL_NAME,
                codec=OpenAIChatCodec(),
                response_codec=OpenAIChatCodec(),
            )
            async for chunk in stream:
                text = stream_text(chunk)
                if text:
                    emitted_output = True
                    yield ImageQueryChunk(text=text)
        except Exception:
            _LOGGER.exception("Image VLM stream failed")
            if not emitted_output:
                yield ImageQueryChunk(text="VLM server unavailable — please retry.")

    async def _stream_vlm(
        self,
        request: nemo_relay.LLMRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        image_uri, query, system_prompt = vision_inputs(request.content)
        image = self.images.resolve(ImageReference(uri=image_uri))
        async for token in self.vlm.stream(
            image,
            query,
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        ):
            yield {"choices": [{"delta": {"content": token}}]}


__all__ = [
    "ImageQueryChunk",
    "ImageQueryRequest",
    "ImageQueryResult",
    "ImageQueryTool",
    "StreamingImageQueryTool",
]
