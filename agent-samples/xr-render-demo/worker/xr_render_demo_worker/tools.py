# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the native tool groups owned by the xr-render worker."""

import asyncio
from pathlib import Path
from typing import Any

from xr_ai_models import VLMService
from xr_ai_tools import ToolSet
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.text_memory import TextMemoryTool
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.video_memory import VideoMemoryTools
from xr_ai_tools.vision import ImageQueryTool, MultiImageQueryTool, VideoQueryTool
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
        self.images = ImageRegistry(allow_external=True)
        self.current_frame = CurrentFrameTool(
            endpoint=frame_endpoint,
            images=self.images,
            frame_max_age_s=frame_max_age_s,
            frame_timeout_s=frame_timeout_s,
        )
        self.image_query = ImageQueryTool(
            images=self.images,
            vlm=vlm,
            system_prompt="Answer directly from the supplied camera image in one short plain-English sentence.",
        )
        self.multi_image_query = MultiImageQueryTool(
            images=self.images,
            vlm=vlm,
            system_prompt="Answer directly from the supplied images in one short plain-English sentence.",
        )
        self.video_query = VideoQueryTool(
            images=self.images,
            vlm=vlm,
            system_prompt="Answer directly from the supplied timed frames in one short plain-English sentence.",
        )
        self.text_memory = TextMemoryTool(text_memory_dir)
        all_tools = (
            *self.scene.tools,
            *self.spatial.tools,
            *self.video.tools,
            self.current_frame,
            self.image_query,
            self.multi_image_query,
            self.video_query,
        )
        self.all = ToolSet(all_tools)
        internal_perception_tools = {
            tool.name
            for tool in (
                *self.video.tools,
                self.current_frame,
                self.image_query,
                self.multi_image_query,
                self.video_query,
            )
        }
        self.model = ToolSet(
            tool
            for tool in all_tools
            if tool.name
            not in internal_perception_tools
            | {
                "start_xr",
                "get_health",
            }
        )

    def release(self, participant_id: str) -> None:
        self.current_frame.release(participant_id)

    async def close(self) -> None:
        await asyncio.gather(
            self.scene.close(),
            self.tracking.close(),
            self.video.close(),
        )


__all__ = ["NativeCapabilities"]
