# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming current-frame vision as a standalone async tool."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import nemo_relay
from nemo_relay.codecs import OpenAIChatCodec
from xr_ai_hub import FrameUnavailable, LiveFrameSource, ProcessorEndpoint
from xr_ai_models import VLMService

from ._pixels import encode_image, frame_to_pil
from ._relay import headers_from_relay
from ._vision import (
    VLM_CALL_NAME,
    VisionChunk,
    VisionRequest,
    openai_response,
    register_frame_sanitizer,
    relay_request,
    stream_text,
    vision_inputs,
)
from .async_tools import AsyncTool

_LOGGER = logging.getLogger(__name__)


class StreamingVisionTool(AsyncTool[VisionRequest, VisionChunk]):
    """A typed current-frame tool that yields answer fragments asynchronously."""

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
        super().__init__(
            "stream_current_frame",
            "Stream an answer about a participant's current live camera view.",
            VisionRequest,
            VisionChunk,
            self._stream_current,
        )

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant disconnects."""

        self.frames.release(participant_id)

    async def _stream_current(
        self,
        request: VisionRequest,
    ) -> AsyncIterator[VisionChunk]:
        try:
            image_url = await self._current_image(request.participant_id)
        except FrameUnavailable as exc:
            yield VisionChunk(text=str(exc))
            return

        await self.endpoint.set_status("processing", request.participant_id)
        fragments: list[str] = []
        emitted_output = False
        try:
            register_frame_sanitizer()
            stream = await nemo_relay.llm.stream_execute(
                VLM_CALL_NAME,
                relay_request(self.system_prompt, image_url, request.query),
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
                    yield VisionChunk(text=text)
        except Exception:
            _LOGGER.exception("Live VLM stream failed")
            if not emitted_output:
                yield VisionChunk(text="VLM server unavailable — please retry.")
        finally:
            await self.endpoint.set_status("idle", request.participant_id)

    async def _current_image(self, participant_id: str) -> str:
        frame = await self.frames.get(participant_id)
        return await asyncio.to_thread(lambda: encode_image(frame_to_pil(frame)))

    async def _stream_vlm(
        self,
        request: nemo_relay.LLMRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        image_url, query, system_prompt = vision_inputs(request.content)
        async for token in self.vlm.stream(
            image_url,
            query,
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        ):
            yield {"choices": [{"delta": {"content": token}}]}


__all__ = ["StreamingVisionTool", "VisionChunk", "VisionRequest"]
