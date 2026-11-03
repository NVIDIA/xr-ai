# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What background model work costs a foreground turn that shares the server.

The tea-making sample sends its monitor's VLM caption and LLM judge call to the same Omni server as the user's
turn. This reproduces that load shape (an image caption plus a judge call on a fixed period) and times a foreground
turn with and without it. Background load is real traffic to the LLM server, so run it with nothing else on it.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

from xr_ai_models import ChatMessage, ImagePart, LLMService, TextPart

from .._gpu import load_tags
from .._media import make_jpeg
from .._prompts import PromptBuilder
from .._store import RunStore
from .._timing import timed_stream

_QUESTION = "In one short sentence, what is the technician doing?"


@dataclass
class ContentionPlan:
    """Foreground sample count and the background load shapes to compare."""

    samples: int = 20
    period_s: float = 2.0
    # name -> (image px, images per call, caption tokens, judge prompt tokens, judge tokens)
    loads: tuple[tuple[str, int, int, int, int, int], ...] = (
        ("idle", 0, 0, 0, 0, 0),
        ("tea_monitor", 672, 1, 128, 600, 40),
        ("heavy_caption", 1280, 2, 180, 600, 40),
    )


def _image_message(px: int, count: int, seed: int) -> ChatMessage:
    parts: list[TextPart | ImagePart] = [
        ImagePart("data:image/jpeg;base64," + base64.b64encode(make_jpeg(px, px, seed=seed + k)).decode())
        for k in range(count)
    ]
    parts.append(TextPart("Describe the scene as compact JSON with fields scene and visible_changes."))
    return ChatMessage(role="user", content=parts)


async def _background(llm: LLMService, builder: PromptBuilder, load: tuple[str, int, int, int, int, int],
                      period_s: float, stop: asyncio.Event) -> int:
    """Caption then judge, once per ``period_s`` (the next cycle starts when the period ends or the work does)."""
    _, px, count, caption_tokens, judge_prompt, judge_tokens = load
    cycles, seed = 0, 1000
    while not stop.is_set():
        started = asyncio.get_running_loop().time()
        await llm.chat([_image_message(px, count, seed)], max_tokens=caption_tokens, enable_thinking=False)
        await llm.chat([builder.user_message(judge_prompt, "Answer yes or no: did the state change?", nonce=True)],
                       max_tokens=judge_tokens, enable_thinking=False)
        cycles += 1
        seed += count
        left = period_s - (asyncio.get_running_loop().time() - started)
        if left > 0:
            try:
                await asyncio.wait_for(stop.wait(), left)
            except TimeoutError:
                pass
    return cycles


async def run(llm: LLMService, store: RunStore, plan: ContentionPlan) -> None:
    """Time a foreground 2k-token turn (TTFT and total) while each background load runs."""
    builder = PromptBuilder()
    await builder.calibrate(llm)
    for load in plan.loads:
        name = load[0]
        stop = asyncio.Event()
        task = asyncio.create_task(_background(llm, builder, load, plan.period_s, stop)) if load[1] else None
        await asyncio.sleep(1.5 if task else 0.0)  # let the background reach steady state
        for _ in range(plan.samples):
            msg = builder.user_message(2_000, _QUESTION, nonce=True)
            t = await timed_stream(llm, [msg], max_tokens=48, enable_thinking=False, temperature=0.0)
            # `foreign` counts requests running while we are idle; our own background load is not foreign here.
            store.add("contention", {"load": name, **load_tags(), "foreign": 0}, **t.as_metrics())
            await asyncio.sleep(0.7)  # foreground turns arrive at human pace, not back to back
        stop.set()
        if task:
            await task
        await asyncio.sleep(2.0)
