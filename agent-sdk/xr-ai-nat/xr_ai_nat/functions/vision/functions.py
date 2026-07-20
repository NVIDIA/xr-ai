# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NAT functions for vision-language model queries."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated, Any

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import ConfigDict, Field

_LOGGER = logging.getLogger(__name__)


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


__all__ = ["VisionFunctionsConfig"]
