# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NAT functions for vision-language model queries."""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any

from nat.plugin_api import (
    Builder,
    FunctionBaseConfig,
    FunctionGroup,
    FunctionGroupBaseConfig,
    FunctionInfo,
    register_function,
    register_function_group,
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from xr_ai_agent import FrameUnavailable, LiveFrameSource

from .._models import _StrictRequest
from ._images import frame_jpeg_data_url

_LOGGER = logging.getLogger(__name__)


class VisionRequest(_StrictRequest):
    """Request a VLM answer from one participant's current camera frame."""

    participant_id: str = Field(description="Participant whose camera frame should be inspected.")
    query: str = Field(min_length=1, description="Question to answer from the camera frame.")


class VisionResult(BaseModel):
    """Complete answer from a live-camera vision invocation."""

    text: str


class VisionChunk(BaseModel):
    """One streamed text fragment from a live-camera vision invocation."""

    text: str


class StreamingVisionConfig(FunctionBaseConfig, name="xr_streaming_vision"):
    """Configure one native streaming function over a live XR camera."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: Any = Field(exclude=True, repr=False)
    vlm: Any = Field(exclude=True, repr=False)
    system_prompt: str = ""
    frame_max_age_s: float = Field(default=2.0, gt=0.0)
    frame_timeout_s: float = Field(default=5.0, gt=0.0)
    _frames: LiveFrameSource | None = PrivateAttr(default=None)

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant disconnects."""

        if self._frames is not None:
            self._frames.release(participant_id)


async def _current_image(frames: LiveFrameSource, participant_id: str) -> str:
    frame = await frames.get(participant_id)
    return await asyncio.to_thread(frame_jpeg_data_url, frame)


@register_function(config_type=StreamingVisionConfig)
async def streaming_vision(config: StreamingVisionConfig, _builder: Builder):
    frames = LiveFrameSource(
        config.endpoint,
        max_age_s=config.frame_max_age_s,
        timeout_s=config.frame_timeout_s,
    )
    config._frames = frames

    async def answer(request: VisionRequest) -> VisionResult:
        await config.endpoint.set_status("processing", request.participant_id)
        try:
            image_url = await _current_image(frames, request.participant_id)
            response = await config.vlm.ask_image(
                image_url,
                request.query,
                system_prompt=config.system_prompt,
            )
            text = (response.content or "").strip()
            if not text:
                text = "I couldn't make out anything in the view."
        except FrameUnavailable as exc:
            text = str(exc)
        except Exception:
            _LOGGER.exception("Live VLM request failed")
            text = "VLM server unavailable — please retry."
        finally:
            await config.endpoint.set_status("idle", request.participant_id)
        return VisionResult(text=text)

    async def stream(request: VisionRequest) -> AsyncGenerator[VisionChunk, None]:
        try:
            image_url = await _current_image(frames, request.participant_id)
        except FrameUnavailable as exc:
            yield VisionChunk(text=str(exc))
            return

        await config.endpoint.set_status("processing", request.participant_id)
        try:
            async for token in config.vlm.stream(
                image_url,
                request.query,
                system_prompt=config.system_prompt,
            ):
                yield VisionChunk(text=token)
        except Exception:
            _LOGGER.exception("Live VLM stream failed")
            yield VisionChunk(text="VLM server unavailable — please retry.")
        finally:
            await config.endpoint.set_status("idle", request.participant_id)

    try:
        yield FunctionInfo.create(
            single_fn=answer,
            stream_fn=stream,
            description="Answer a question about a participant's current live camera view.",
        )
    finally:
        config._frames = None


class VisionFunctionsConfig(FunctionGroupBaseConfig, name="xr_vision"):
    """Configure image-question answering over an injected VLM service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vlm: Any = Field(exclude=True, repr=False)
    system_prompt: str = ""


@register_function_group(config_type=VisionFunctionsConfig)
async def vision_functions(config: VisionFunctionsConfig, _builder: Builder):
    group = FunctionGroup(config=config)

    async def ask_image(
        question: Annotated[str, Field(description="Question to answer from the acquired image.")],
        image_path: Annotated[
            str,
            Field(
                description=(
                    "Absolute local PNG or JPEG path returned by image acquisition. "
                    "Never invent or guess a path."
                )
            ),
        ],
    ) -> str:
        if not image_path:
            return "ask_image: image_path is empty — acquire an image first."
        path = Path(image_path)
        if not path.exists():
            return f"ask_image: file not found at {image_path!r}."

        from ._images import load_jpeg_data_url

        try:
            data_url = await asyncio.to_thread(load_jpeg_data_url, path)
        except Exception as exc:
            _LOGGER.exception("Failed to load image at %s", image_path)
            return f"ask_image: failed to read image at {image_path!r}: {exc}"

        import httpx

        try:
            response = await config.vlm.ask_image(
                data_url,
                question,
                system_prompt=config.system_prompt,
            )
        except httpx.HTTPError as exc:
            _LOGGER.exception("VLM request failed")
            return f"ask_image: vlm-server request failed: {exc}"

        content = re.sub(
            r"<think>.*?</think>",
            "",
            response.content,
            flags=re.DOTALL,
        ).strip()
        return content

    group.add_function(
        "ask_image",
        ask_image,
        description=(
            "Ask a vision-language model a question about an acquired local image file. "
            "First acquire an image through the appropriate live or recorded-frame capability, "
            "then pass its exact returned path. Never invent or guess an image path."
        ),
    )

    yield group


__all__ = [
    "StreamingVisionConfig",
    "VisionChunk",
    "VisionFunctionsConfig",
    "VisionRequest",
    "VisionResult",
]
