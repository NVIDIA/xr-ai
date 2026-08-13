# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the native tool groups owned by the xr-render worker."""

import asyncio
from pathlib import Path
from typing import Any

from xr_ai_models import VLMService
from xr_ai_tools import ToolSet
from xr_ai_tools.historical_vision import HistoricalVisionTool
from xr_ai_tools.live_vision import LiveVisionTool
from xr_ai_tools.text_memory import TextMemoryTool
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.video_memory import VideoMemoryTools
from xr_render_scene import SceneTools

from .spatial_tools import RenderSpatialTools


class NativeCapabilities:
    """Own every native tool and service client used by the render worker."""

    def __init__(
        self,
        *,
        scene_endpoint: str,
        openxr_endpoint: str,
        video_memory_endpoint: str,
        frame_endpoint: Any,
        vlm: VLMService,
        text_memory_dir: str | Path,
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        self.scene = SceneTools(scene_endpoint)
        self.tracking = TrackingTools(openxr_endpoint)
        self.spatial = RenderSpatialTools(self.tracking)
        self.video = VideoMemoryTools(video_memory_endpoint)
        self.live_vision = LiveVisionTool(
            endpoint=frame_endpoint,
            vlm=vlm,
            system_prompt="Answer directly from the visible camera image in one short plain-English sentence.",
            frame_max_age_s=frame_max_age_s,
            frame_timeout_s=frame_timeout_s,
            manage_status=False,
        )
        self.past_vision = HistoricalVisionTool(video=self.video, vlm=vlm)
        self.text_memory = TextMemoryTool(text_memory_dir)
        all_tools = (
            *self.scene.tools,
            *self.spatial.tools,
            *self.video.tools,
            self.live_vision,
            self.past_vision,
        )
        self.all = ToolSet(all_tools)
        self.model = ToolSet(
            tool
            for tool in all_tools
            if tool.name not in {"start_xr", "get_health"}
        )

    def release(self, participant_id: str) -> None:
        self.live_vision.release(participant_id)

    async def close(self) -> None:
        await asyncio.gather(
            self.scene.close(),
            self.tracking.close(),
            self.video.close(),
        )


__all__ = ["NativeCapabilities"]
