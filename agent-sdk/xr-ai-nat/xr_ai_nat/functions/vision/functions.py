# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NAT functions for current and recorded camera frames."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

from nat.plugin_api import (
    Builder,
    FunctionBaseConfig,
    FunctionGroup,
    FunctionGroupBaseConfig,
    FunctionGroupRef,
    FunctionInfo,
    register_function,
    register_function_group,
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from xr_ai_agent import FrameUnavailable, LiveFrameSource

from .._models import _StrictRequest

_LOGGER = logging.getLogger(__name__)


class VisionRequest(_StrictRequest):
    """Request a VLM answer from one participant's current camera frame."""

    participant_id: str = Field(description="Participant whose camera frame should be inspected.")
    query: str = Field(min_length=1, description="Question to answer from the camera frame.")


class VisionResult(BaseModel):
    """Complete answer from a live-camera vision invocation."""

    text: str = Field(description="Answer text or a reason that vision is unavailable.")
    status: Literal["ok", "unavailable"] = Field(
        default="ok",
        description=(
            "Whether a current frame and VLM answer were available. "
            "Callers must handle an unavailable result without interpreting its text as a scene answer."
        ),
    )


class VisionChunk(BaseModel):
    """One streamed text fragment from a live-camera vision invocation."""

    text: str = Field(description="A partial fragment of the streamed answer text.")


class LiveVisionRequest(_StrictRequest):
    """Ask a question about a participant's present live camera frame."""

    participant_id: str = Field(description="Participant whose camera should be examined.")
    question: str = Field(min_length=1, description="Specific question about the live camera frame.")


class LiveVisionResult(BaseModel):
    """Answer produced from a current or recorded camera frame."""

    answer: str = Field(description="Plain-English answer derived from the camera frame.")


class HistoricalVisionRequest(_StrictRequest):
    """Ask a question about a recorded camera frame from the recent past."""

    participant_id: str = Field(description="Participant whose recorded camera frame should be inspected.")
    question: str = Field(min_length=1, description="Specific question about the recorded camera frame.")
    second_ago: int = Field(gt=0, description="Positive offset from the utterance time in seconds.")
    reference_time_us: int = Field(
        gt=0,
        description="Reference (utterance) time in microseconds that the offset counts back from.",
    )


async def _current_image(frames: LiveFrameSource, participant_id: str) -> str:
    from ._pixels import encode_image, frame_to_pil

    frame = await frames.get(participant_id)
    return await asyncio.to_thread(lambda: encode_image(frame_to_pil(frame)))


async def _ask_image(
    vlm: Any,
    image: str | Path,
    question: str,
    system_prompt: str,
    unavailable_message: str,
) -> str:
    response = await vlm.ask_image(image, question, system_prompt=system_prompt)
    text = (response.content or "").strip()
    if not text:
        raise FrameUnavailable(unavailable_message)
    return text


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
        status: Literal["ok", "unavailable"] = "ok"
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
            status = "unavailable"
        except Exception:
            _LOGGER.exception("Live VLM request failed")
            text = "VLM server unavailable — please retry."
            status = "unavailable"
        finally:
            await config.endpoint.set_status("idle", request.participant_id)
        return VisionResult(text=text, status=status)

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


class VisionToolsConfig(FunctionGroupBaseConfig, name="xr_vision_tools"):
    """Configure the one-shot vision tools used by agent workflows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: Any = Field(exclude=True, repr=False)
    vlm: Any = Field(exclude=True, repr=False)
    frame_max_age_s: float = Field(default=2.0, gt=0.0)
    frame_timeout_s: float = Field(default=5.0, gt=0.0)
    video_memory: FunctionGroupRef = FunctionGroupRef("video_memory")
    _frames: LiveFrameSource | None = PrivateAttr(default=None)

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant leaves."""

        if self._frames is not None:
            self._frames.release(participant_id)


@register_function_group(config_type=VisionToolsConfig)
async def vision_tools(config: VisionToolsConfig, builder: Builder):
    """Build current and historical vision tools over one live frame source."""
    frames = LiveFrameSource(
        config.endpoint,
        max_age_s=config.frame_max_age_s,
        timeout_s=config.frame_timeout_s,
    )
    config._frames = frames

    # The recorded-frame dependency is resolved lazily so a live-only consumer can
    # build and call `look_at_current_frame` without configuring a video-memory group.
    # `look_at_past_frame` resolves (and caches) the group on first invocation.
    recorded_frame: Any = None

    async def _get_recorded_frame() -> Any:
        nonlocal recorded_frame
        if recorded_frame is None:
            video_memory = await builder.get_function_group(config.video_memory)
            recorded_functions = await video_memory.get_all_functions()
            recorded_frame = recorded_functions[
                f"{video_memory.instance_name}__get_frame_from_time"
            ]
        return recorded_frame

    async def look(request: LiveVisionRequest) -> LiveVisionResult:
        text = await _ask_image(
            config.vlm,
            await _current_image(frames, request.participant_id),
            request.question,
            "Answer directly from the visible camera image in one short plain-English sentence.",
            "The camera image did not produce an answer.",
        )
        return LiveVisionResult(answer=text)

    async def look_past(request: HistoricalVisionRequest) -> LiveVisionResult:
        recorded_frame = await _get_recorded_frame()
        frame = await recorded_frame.ainvoke(
            {
                "participant_id": request.participant_id,
                "second_ago": request.second_ago,
                "reference_time_us": request.reference_time_us,
            }
        )
        text = await _ask_image(
            config.vlm,
            Path(frame.path),
            request.question,
            "Answer directly from this recorded camera frame in one short sentence.",
            "The recorded camera image did not produce an answer.",
        )
        return LiveVisionResult(answer=text)

    group = FunctionGroup(config=config)
    group.add_function(
        "look_at_current_frame",
        look,
        description=(
            "Inspect the user's present physical view when a request explicitly requires a visible fact. "
            "Do not use this tool to interpret conversation or inspect the virtual XR scene."
        ),
    )
    group.add_function(
        "look_at_past_frame",
        look_past,
        description=(
            "Inspect a recorded camera frame only for an explicitly historical question, using a positive "
            "seconds offset from the user's utterance time."
        ),
    )
    try:
        yield group
    finally:
        config._frames = None


__all__ = [
    "HistoricalVisionRequest",
    "LiveVisionRequest",
    "LiveVisionResult",
    "StreamingVisionConfig",
    "VisionChunk",
    "VisionRequest",
    "VisionResult",
    "VisionToolsConfig",
]
