# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM latency against image count and resolution, streaming and non-streaming."""

from __future__ import annotations

from dataclasses import dataclass

from xr_ai_models import VLMService

from .._gpu import load_tags
from .._media import make_jpeg
from .._store import RunStore
from .._timing import now_ms

RESOLUTIONS = ((336, 336), (672, 672), (1280, 720), (1920, 1080))
IMAGE_COUNTS = (1, 2, 4)
_QUESTION = "Describe what you see in one short sentence."


@dataclass
class VlmPlan:
    """Sample count and sweeps; the live server allows at most 4 images per prompt."""

    samples: int = 10
    resolutions: tuple[tuple[int, int], ...] = RESOLUTIONS
    counts: tuple[int, ...] = IMAGE_COUNTS


async def run(vlm: VLMService, store: RunStore, plan: VlmPlan) -> None:
    """Time chat and streamed first-text for each (resolution, image count) cell.

    Every cold call gets images no earlier call has seen, so the server's prefix cache cannot answer it. A final
    ``cached`` pair repeats one chat request exactly, which is what asking twice about the same frame costs.
    """
    seed = 0

    def fresh(width: int, height: int, count: int) -> list[bytes]:
        nonlocal seed
        images = [make_jpeg(width, height, seed=seed + k) for k in range(count)]
        seed += count
        return images

    for width, height in plan.resolutions:
        for count in plan.counts:
            for _ in range(plan.samples):
                images = fresh(width, height, count)
                case = {"res": f"{width}x{height}", "images": count, "jpeg_kb": sum(map(len, images)) // 1024,
                        **load_tags()}
                start = now_ms()
                response = await vlm.ask_images(images, _QUESTION, max_tokens=32, temperature=0.0)
                total = now_ms() - start
                usage = response.raw.get("usage") or {}
                store.add("vlm.chat", case, total=total, prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"))
                images = fresh(width, height, count)
                start = now_ms()
                first = None
                async for _ in vlm.stream_images(images, _QUESTION, max_tokens=32, temperature=0.0):
                    if first is None:
                        first = now_ms() - start
                store.add("vlm.stream", case, first_text=first, total=now_ms() - start)
            images = fresh(width, height, count)
            await vlm.ask_images(images, _QUESTION, max_tokens=32, temperature=0.0)
            for _ in range(plan.samples):
                start = now_ms()
                await vlm.ask_images(images, _QUESTION, max_tokens=32, temperature=0.0)
                store.add("vlm.cached", {"res": f"{width}x{height}", "images": count, **load_tags()},
                          total=now_ms() - start)
