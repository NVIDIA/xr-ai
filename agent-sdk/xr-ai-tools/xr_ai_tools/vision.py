# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM tools over single images, image collections, and timed video frames."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import nemo_relay
from nemo_relay.codecs import OpenAIChatCodec
from pydantic import BaseModel, Field, model_validator
from xr_ai_models import VLMService

from ._relay import headers_from_relay
from ._vision import (
    VLM_CALL_NAME,
    image_sanitizer,
    openai_response,
    relay_request,
    response_text,
    stream_text,
    timestamped_question,
    visible_text,
    vision_inputs,
)
from .async_tools import AsyncTool
from .image import ImageReference, ImageRegistry, TimedImage
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)
_MAX_IMAGES = 4


class ImageQueryRequest(StrictRequest):
    """An image reference and the question to answer from it."""

    image: ImageReference = Field(description="Image selected by another tool or caller.")
    """Image selected by another tool or caller."""

    query: str = Field(min_length=1, description="Question to answer from the image.")
    """Question to answer from the image."""


class MultiImageQueryRequest(StrictRequest):
    """An ordered image collection and one question spanning the images."""

    images: list[ImageReference] = Field(
        min_length=1,
        max_length=_MAX_IMAGES,
        description="Ordered images selected by other tools or the caller.",
    )
    """Ordered images selected by other tools or the caller."""

    query: str = Field(min_length=1, description="Question to answer from the images.")
    """Question to answer from the images."""


class VideoQueryRequest(StrictRequest):
    """Chronological image frames and a temporal question about them."""

    frames: list[TimedImage] = Field(
        min_length=1,
        max_length=_MAX_IMAGES,
        description="Chronologically ordered image frames with Unix timestamps.",
    )
    """Chronologically ordered image frames with Unix timestamps."""

    query: str = Field(min_length=1, description="Question to answer from the timed frames.")
    """Question to answer from the timed frames."""

    @model_validator(mode="after")
    def validate_timeline(self) -> VideoQueryRequest:
        """Require frames to be supplied in chronological order."""

        timestamps = [frame.timestamp_us for frame in self.frames]
        if timestamps != sorted(timestamps):
            raise ValueError("frames must be in chronological timestamp order")
        return self


class ImageQueryResult(BaseModel):
    """A complete VLM answer and visual-input availability state."""

    text: str = Field(description="Complete answer text.")
    """Complete answer text."""

    available: bool = Field(
        default=True,
        description="Whether the supplied visual input produced a usable answer.",
    )
    """Whether the supplied visual input produced a usable answer."""


class ImageQueryChunk(BaseModel):
    """One incremental text fragment from a streaming VLM answer."""

    text: str = Field(description="A partial fragment of the streamed answer text.")
    """Partial fragment of the streamed answer text."""


class _ImageInference:
    """Run all image cardinalities through one Relay and VLM path."""

    def __init__(self, images: ImageRegistry, vlm: VLMService, system_prompt: str) -> None:
        self.images = images
        self.vlm = vlm
        self.system_prompt = system_prompt

    async def answer(
        self,
        inputs: Sequence[tuple[ImageReference, int | None]],
        query: str,
    ) -> ImageQueryResult:
        try:
            for image, _timestamp_us in inputs:
                self.images.resolve(image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Image input could not be resolved: %s", error)
            return ImageQueryResult(
                text="Image input unavailable — please select it again.",
                available=False,
            )
        try:
            with image_sanitizer():
                response = await nemo_relay.llm.execute(
                    VLM_CALL_NAME,
                    relay_request(
                        self.system_prompt,
                        [(image.uri, timestamp_us) for image, timestamp_us in inputs],
                        query,
                    ),
                    self._ask_vlm,
                    model_name=VLM_CALL_NAME,
                    codec=OpenAIChatCodec(),
                    response_codec=OpenAIChatCodec(),
                )
            text = visible_text(response_text(response))
            if not text:
                return ImageQueryResult(
                    text="The visual input did not produce an answer.",
                    available=False,
                )
            return ImageQueryResult(text=text)
        except Exception:
            _LOGGER.exception("Image VLM request failed")
            return ImageQueryResult(
                text="VLM server unavailable — please retry.",
                available=False,
            )

    async def stream(
        self,
        inputs: Sequence[tuple[ImageReference, int | None]],
        query: str,
    ) -> AsyncIterator[ImageQueryChunk]:
        fragments: list[str] = []
        emitted_output = False
        try:
            for image, _timestamp_us in inputs:
                self.images.resolve(image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Image input could not be resolved: %s", error)
            yield ImageQueryChunk(text="Image input unavailable — please select it again.")
            return
        try:
            started_at = time.monotonic()
            _LOGGER.info("Image VLM stream request started image_count=%d", len(inputs))
            with image_sanitizer():
                stream = await nemo_relay.llm.stream_execute(
                    VLM_CALL_NAME,
                    relay_request(
                        self.system_prompt,
                        [(image.uri, timestamp_us) for image, timestamp_us in inputs],
                        query,
                    ),
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
                        if not emitted_output:
                            _LOGGER.info(
                                "Image VLM stream first token latency_ms=%.1f image_count=%d",
                                (time.monotonic() - started_at) * 1000,
                                len(inputs),
                            )
                        emitted_output = True
                        yield ImageQueryChunk(text=text)
        except Exception:
            _LOGGER.exception("Image VLM stream failed")
            if not emitted_output:
                yield ImageQueryChunk(text="VLM server unavailable — please retry.")

    async def _ask_vlm(self, request: nemo_relay.LLMRequest) -> dict[str, Any]:
        inputs, query, system_prompt = vision_inputs(request.content)
        response = await self.vlm.ask_images(
            [self.images.resolve(ImageReference(uri=uri)) for uri, _timestamp in inputs],
            timestamped_question(query, [timestamp for _uri, timestamp in inputs]),
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        )
        return openai_response(response.content)

    async def _stream_vlm(
        self,
        request: nemo_relay.LLMRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        inputs, query, system_prompt = vision_inputs(request.content)
        async for token in self.vlm.stream_images(
            [self.images.resolve(ImageReference(uri=uri)) for uri, _timestamp in inputs],
            timestamped_question(query, [timestamp for _uri, timestamp in inputs]),
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        ):
            yield {"choices": [{"delta": {"content": token}}]}


class ImageQueryTool(Tool[ImageQueryRequest, ImageQueryResult]):
    """Answer a question about one caller-selected image."""

    def __init__(self, *, images: ImageRegistry, vlm: VLMService, system_prompt: str = "") -> None:
        inference = _ImageInference(images, vlm, system_prompt)
        super().__init__(
            "query_image",
            "Answer a question about one image reference returned by an image tool or supplied by the caller.",
            ImageQueryRequest,
            ImageQueryResult,
            lambda request: inference.answer(((request.image, None),), request.query),
            render_result=lambda result: result.text,
        )


class MultiImageQueryTool(Tool[MultiImageQueryRequest, ImageQueryResult]):
    """Answer one question across an ordered image collection."""

    def __init__(self, *, images: ImageRegistry, vlm: VLMService, system_prompt: str = "") -> None:
        inference = _ImageInference(images, vlm, system_prompt)
        super().__init__(
            "query_images",
            "Answer a question across multiple ordered image references.",
            MultiImageQueryRequest,
            ImageQueryResult,
            lambda request: inference.answer(
                tuple((image, None) for image in request.images),
                request.query,
            ),
            render_result=lambda result: result.text,
        )


class VideoQueryTool(Tool[VideoQueryRequest, ImageQueryResult]):
    """Answer one temporal question across timestamped image frames."""

    def __init__(self, *, images: ImageRegistry, vlm: VLMService, system_prompt: str = "") -> None:
        inference = _ImageInference(images, vlm, system_prompt)
        super().__init__(
            "query_video",
            "Answer a temporal question across chronologically ordered image frames and timestamps.",
            VideoQueryRequest,
            ImageQueryResult,
            lambda request: inference.answer(
                tuple((frame.image, frame.timestamp_us) for frame in request.frames),
                request.query,
            ),
            render_result=lambda result: result.text,
        )


class StreamingImageQueryTool(AsyncTool[ImageQueryRequest, ImageQueryChunk]):
    """Stream an answer about one caller-selected image."""

    def __init__(self, *, images: ImageRegistry, vlm: VLMService, system_prompt: str = "") -> None:
        inference = _ImageInference(images, vlm, system_prompt)
        super().__init__(
            "stream_image_query",
            "Stream an answer about one image reference returned by an image tool or supplied by the caller.",
            ImageQueryRequest,
            ImageQueryChunk,
            lambda request: inference.stream(((request.image, None),), request.query),
        )


__all__ = [
    "ImageQueryChunk",
    "ImageQueryRequest",
    "ImageQueryResult",
    "ImageQueryTool",
    "MultiImageQueryRequest",
    "MultiImageQueryTool",
    "StreamingImageQueryTool",
    "VideoQueryRequest",
    "VideoQueryTool",
]
