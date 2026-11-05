# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM latency against prompt size, cache state, output length, and thinking mode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from xr_ai_models import ChatMessage, LLMService

from .._gpu import load_tags
from .._prompts import PromptBuilder
from .._store import RunStore
from .._timing import timed_chat, timed_stream

CONTEXT_TOKENS = (128, 512, 2_000, 4_000, 8_000, 16_000, 28_000)
OUTPUT_TOKENS = (8, 32, 128, 512)
_QUESTION = "In one short sentence, what is the technician doing?"
_COUNT = "Count upward from 1 separated by commas. Output only the numbers and never stop."


@dataclass
class LlmPlan:
    """Sample counts and sweeps; defaults keep total runtime to a few minutes of light load."""

    samples: int = 12
    contexts: tuple[int, ...] = CONTEXT_TOKENS
    outputs: tuple[int, ...] = OUTPUT_TOKENS
    concurrency: tuple[int, ...] = (1, 2, 4, 8)
    skip: set[str] = field(default_factory=set)


async def run(llm: LLMService, store: RunStore, plan: LlmPlan, live: LLMService | None = None) -> None:
    """Run every LLM sweep on ``llm``, appending samples to ``store``.

    ``llm`` must not cap output or temperature in its profile: ``default_extras`` override per-call arguments.
    ``live`` is the profile as an application gets it; it only runs the live-defaults case.
    """
    builder = PromptBuilder()
    store.write_meta({"tokens_per_char": await builder.calibrate(llm)})
    if "context" not in plan.skip:
        await _context_sweep(llm, store, plan, builder)
    if "prefix" not in plan.skip:
        await _prefix_cache(llm, store, plan, builder)
    if "output" not in plan.skip:
        await _output_sweep(llm, store, plan)
    if "thinking" not in plan.skip:
        await _thinking(llm, store, plan, builder)
    if "concurrency" not in plan.skip:
        await _concurrency(llm, store, plan, builder)
    if live is not None and "live" not in plan.skip:
        await _live_defaults(live, store, plan, builder)


def _case(**kw: object) -> dict[str, object]:
    return {**kw, **load_tags()}


async def _context_sweep(llm: LLMService, store: RunStore, plan: LlmPlan, b: PromptBuilder) -> None:
    """Cold prefill: TTFT of visible text vs prompt size, thinking off, unique prefix every call."""
    for tokens in plan.contexts:
        for _ in range(plan.samples):
            msg = b.user_message(tokens, _QUESTION, nonce=True)
            t = await timed_stream(llm, [msg], max_tokens=24, enable_thinking=False, temperature=0.0)
            store.add("llm.context.cold", _case(ctx_tokens=tokens, mode="stream", thinking=False), **t.as_metrics())
        # Non-streaming gives exact server token counts for the same size.
        msg = b.user_message(tokens, _QUESTION, nonce=True)
        c = await timed_chat(llm, [msg], max_tokens=24, enable_thinking=False, temperature=0.0)
        store.add("llm.context.cold", _case(ctx_tokens=tokens, mode="chat", thinking=False), **c.as_metrics())


async def _prefix_cache(llm: LLMService, store: RunStore, plan: LlmPlan, b: PromptBuilder) -> None:
    """Warm prefix: identical long prefix, only the final question changes (the conversation shape)."""
    for tokens in plan.contexts:
        prefix = b.filler(tokens, nonce=True)
        for i in range(plan.samples + 1):
            msg = ChatMessage(role="user", content=f"{prefix}\n\n{_QUESTION} (variant {i})")
            t = await timed_stream(llm, [msg], max_tokens=24, enable_thinking=False, temperature=0.0)
            # Call 0 populates the cache and is recorded as cold; the rest are warm.
            store.add("llm.context.prefix", _case(ctx_tokens=tokens, warm=i > 0), **t.as_metrics())


async def _output_sweep(llm: LLMService, store: RunStore, plan: LlmPlan) -> None:
    """Decode rate and total time vs output length, short prompt, thinking off."""
    msg = ChatMessage(role="user", content=_COUNT)
    for n in plan.outputs:
        for _ in range(plan.samples):
            t = await timed_stream(llm, [msg], max_tokens=n, enable_thinking=False, temperature=0.0)
            store.add("llm.output", _case(max_tokens=n, mode="stream"), **t.as_metrics())
            c = await timed_chat(llm, [msg], max_tokens=n, enable_thinking=False, temperature=0.0)
            store.add("llm.output", _case(max_tokens=n, mode="chat"), **c.as_metrics())


async def _thinking(llm: LLMService, store: RunStore, plan: LlmPlan, b: PromptBuilder) -> None:
    """Time to first reasoning token vs first visible token, thinking on and off, at 2k context."""
    for thinking in (False, True):
        for _ in range(plan.samples):
            msg = b.user_message(2_000, _QUESTION, nonce=True)
            t = await timed_stream(
                llm, [msg], max_tokens=1024, enable_thinking=thinking, temperature=0.0,
            )
            store.add("llm.thinking", _case(ctx_tokens=2_000, thinking=thinking), **t.as_metrics())


async def _concurrency(llm: LLMService, store: RunStore, plan: LlmPlan, b: PromptBuilder) -> None:
    """N simultaneous 2k-context requests: TTFT and total vs queue depth."""
    for width in plan.concurrency:
        for _ in range(max(plan.samples // 3, 3)):
            msgs = [b.user_message(2_000, _QUESTION, nonce=True) for _ in range(width)]
            results = await asyncio.gather(
                *(timed_stream(llm, [m], max_tokens=48, enable_thinking=False, temperature=0.0) for m in msgs)
            )
            for t in results:
                store.add("llm.concurrency", _case(width=width), **t.as_metrics())


async def _live_defaults(llm: LLMService, store: RunStore, plan: LlmPlan, b: PromptBuilder) -> None:
    """The profile exactly as applications get it: no per-call overrides, so its defaults decide everything.

    Also records whether a per-call ``max_tokens`` is honoured (it is not when ``default_extras`` sets one).
    """
    for _ in range(plan.samples):
        msg = b.user_message(2_000, _QUESTION, nonce=True)
        t = await timed_stream(llm, [msg])
        store.add("llm.live_defaults", _case(ctx_tokens=2_000, call="no_overrides"), **t.as_metrics())
        c = await timed_chat(llm, [b.user_message(2_000, _QUESTION, nonce=True)], max_tokens=16, enable_thinking=False)
        store.add("llm.live_defaults", _case(ctx_tokens=2_000, call="max_tokens_16_thinking_off"), **c.as_metrics())
