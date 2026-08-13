# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite current-frame vision for ordinary agent tool calls."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import nemo_relay
from nemo_relay.codecs import OpenAIChatCodec
from xr_ai_hub import FrameUnavailable, LiveFrameSource, ProcessorEndpoint
from xr_ai_models import VLMService

from ._pixels import encode_image, frame_to_pil
from ._relay import headers_from_relay
from ._vision import (
    VLM_CALL_NAME,
    VisionRequest,
    VisionResponse,
    openai_response,
    register_frame_sanitizer,
    relay_request,
    response_text,
    vision_inputs,
)
from .tools import Tool

_LOGGER = logging.getLogger(__name__)


class LiveVisionTool(Tool[VisionRequest, VisionResponse]):
    """A finite current-frame tool for agentic planning and tool loops."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        vlm: VLMService,
        system_prompt: str = "",
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
        manage_status: bool = True,
    ) -> None:
        if frame_max_age_s <= 0.0:
            raise ValueError("frame_max_age_s must be positive")
        if frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")
        self.endpoint = endpoint
        self.vlm = vlm
        self.system_prompt = system_prompt
        self.manage_status = manage_status
        self.frames = LiveFrameSource(
            endpoint,
            max_age_s=frame_max_age_s,
            timeout_s=frame_timeout_s,
        )
        super().__init__(
            "look_at_current_frame",
            "Answer a question about a participant's current live camera view.",
            VisionRequest,
            VisionResponse,
            self._answer_current,
            render_result=lambda result: result.text,
        )

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant disconnects."""

        self.frames.release(participant_id)

    async def _answer_current(self, request: VisionRequest) -> VisionResponse:
        try:
            image_url = await self._current_image(request.participant_id)
        except FrameUnavailable as exc:
            return VisionResponse(text=str(exc))

        if self.manage_status:
            await self.endpoint.set_status("processing", request.participant_id)
        try:
            register_frame_sanitizer()
            response = await nemo_relay.llm.execute(
                VLM_CALL_NAME,
                relay_request(self.system_prompt, image_url, request.query),
                self._ask_vlm,
                model_name=VLM_CALL_NAME,
                codec=OpenAIChatCodec(),
                response_codec=OpenAIChatCodec(),
            )
            return VisionResponse(text=response_text(response))
        except Exception:
            _LOGGER.exception("Live VLM request failed")
            return VisionResponse(text="VLM server unavailable — please retry.")
        finally:
            if self.manage_status:
                await self.endpoint.set_status("idle", request.participant_id)

    async def _current_image(self, participant_id: str) -> str:
        frame = await self.frames.get(participant_id)
        return await asyncio.to_thread(lambda: encode_image(frame_to_pil(frame)))

    async def _ask_vlm(self, request: nemo_relay.LLMRequest) -> dict[str, Any]:
        image_url, query, system_prompt = vision_inputs(request.content)
        text = await self.vlm.ask_image(
            image_url,
            query,
            system_prompt=system_prompt,
            headers=headers_from_relay(request.headers),
        )
        return openai_response(text.content)


__all__ = ["LiveVisionTool", "VisionRequest", "VisionResponse"]
