# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-based image-question tool served by vlm-mcp.

vlm-mcp is a pure file → VLM wrapper: it owns no live camera or recorded
history, so it cannot use the native always-on ``xr_vision_tools`` surface. The
path-based ``ask_image`` tool therefore lives with its only consumer.

This module deliberately avoids ``from __future__ import annotations`` so NAT
can introspect the ``Annotated`` tool signature to build its input schema.
"""

import asyncio
import re
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
from nat.plugin_api import (
    Builder,
    FunctionGroup,
    FunctionGroupBaseConfig,
    register_function_group,
)
from pydantic import ConfigDict, Field


def load_jpeg_data_url(image_path: str | Path, quality: int = 85) -> str:
    """Convert a local image to an RGB JPEG data URL."""

    import base64
    import io

    from PIL import Image

    with Image.open(image_path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class AskImageConfig(FunctionGroupBaseConfig, name="xr_vlm_ask_image"):
    """Configure the file-based image-question tool served over MCP."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vlm: Any = Field(exclude=True, repr=False)
    system_prompt: str = ""


@register_function_group(config_type=AskImageConfig)
async def ask_image_group(config: AskImageConfig, _builder: Builder):
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

        try:
            data_url = await asyncio.to_thread(load_jpeg_data_url, path)
        except Exception as exc:
            logger.exception("Failed to load image at {}", image_path)
            return f"ask_image: failed to read image at {image_path!r}: {exc}"

        import httpx

        try:
            response = await config.vlm.ask_image(
                data_url,
                question,
                system_prompt=config.system_prompt,
            )
        except httpx.HTTPError as exc:
            logger.exception("VLM request failed")
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
