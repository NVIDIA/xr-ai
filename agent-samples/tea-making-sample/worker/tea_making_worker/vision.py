# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-frame VLM observation loop for YAML workflow steps."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger
from xr_ai_hub import FrameUnavailable, LiveFrameSource
from xr_ai_nat.functions.vision._pixels import encode_image, frame_to_pil

from .workflow import WorkflowStep, render_template


@dataclass(frozen=True, slots=True)
class VisualObservation:
    """One VLM caption tied to a live frame timestamp."""

    text: str
    frame_pts_us: int


class StepVision:
    """Run a step-specific VLM prompt over the participant's current frame."""

    def __init__(
        self,
        *,
        endpoint: Any,
        vlm: Any,
        frame_max_age_s: float,
        frame_timeout_s: float,
        vlm_timeout_s: float = 15.0,
        system_prompt: str = "",
    ) -> None:
        self._frames = LiveFrameSource(
            endpoint,
            max_age_s=frame_max_age_s,
            timeout_s=frame_timeout_s,
        )
        self._vlm = vlm
        self._frame_timeout_s = frame_timeout_s
        self._vlm_timeout_s = vlm_timeout_s
        self._system_prompt = system_prompt

    async def observe(
        self,
        participant_id: str,
        step: WorkflowStep,
        context: dict[str, Any],
        *,
        task: dict[str, Any],
    ) -> VisualObservation:
        started = time.perf_counter()
        logger.debug(
            "vlm frame wait begin pid={} step={} timeout_s={}",
            participant_id,
            step.id,
            self._frame_timeout_s,
        )
        try:
            async with asyncio.timeout(self._frame_timeout_s + 1.0):
                frame = await self._frames.get(participant_id)
        except TimeoutError as exc:
            raise FrameUnavailable("Camera frame request timed out.") from exc
        frame_elapsed_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "vlm frame acquired pid={} step={} frame_pts_us={} elapsed_ms={:.1f}",
            participant_id,
            step.id,
            frame.pts_us,
            frame_elapsed_ms,
        )
        image = encode_image(frame_to_pil(frame))
        prompt = render_template(
            step.vlm_prompt,
            context=context,
            step=step,
            task=task,
        )
        vlm_started = time.perf_counter()
        logger.debug(
            "vlm request begin pid={} step={} timeout_s={}",
            participant_id,
            step.id,
            self._vlm_timeout_s,
        )
        response = await self._vlm.ask_image(
            image,
            prompt,
            system_prompt=self._system_prompt,
            temperature=0.0,
            timeout=self._vlm_timeout_s,
        )
        logger.debug(
            "vlm request complete pid={} step={} request_ms={:.1f} total_ms={:.1f}",
            participant_id,
            step.id,
            (time.perf_counter() - vlm_started) * 1000,
            (time.perf_counter() - started) * 1000,
        )
        return VisualObservation(text=(response.content or "").strip(), frame_pts_us=frame.pts_us)

    def release(self, participant_id: str) -> None:
        self._frames.release(participant_id)


__all__ = ["FrameUnavailable", "StepVision", "VisualObservation"]
