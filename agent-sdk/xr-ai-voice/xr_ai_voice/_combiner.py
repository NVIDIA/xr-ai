# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-backed combination of pending spoken updates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

from loguru import logger
from xr_ai_models import ChatMessage, LLMService, ModelServiceError

_DEFAULT_PROMPT = """Combine simultaneous spoken updates into one concise, natural response.
Preserve every actionable fact, proper name, value, unit, and warning. Remove repetition.
Do not invent information or mention agents, queues, sources, or aggregation.
Use plain spoken prose with at most two sentences."""


def llm_update_combiner(
    llm: LLMService,
    *,
    prompt: str = _DEFAULT_PROMPT,
    max_tokens: int = 192,
    timeout_s: float = 5.0,
) -> Callable[[Sequence[str]], Awaitable[str]]:
    """Return a ``VoiceAgent`` ``combiner`` that rewrites pending updates with ``llm``.

    The combiner rewrites several updates into one spoken utterance. It uses
    temperature 0 with thinking disabled. It falls back to joining the updates
    with spaces, and logs why, when the model service fails, the call times
    out, the model returns no text or tool calls, or the rewrite stops at the
    token limit. A single update is returned unchanged without a model call.

    :param llm: Model service that rewrites the updates.
    :param prompt: System prompt for the rewrite.
    :param max_tokens: Token limit for the rewrite.
    :param timeout_s: Deadline for one rewrite call, in seconds.
    :raises ValueError: If ``prompt`` is empty or ``max_tokens`` or ``timeout_s`` is not positive.
    """

    if not prompt.strip():
        raise ValueError("update combiner prompt must not be empty")
    if max_tokens <= 0:
        raise ValueError("update combiner token limit must be positive")
    if timeout_s <= 0:
        raise ValueError("update combiner timeout must be positive")
    system_prompt = prompt.strip()

    async def combine(texts: Sequence[str]) -> str:
        updates = [text.strip() for text in texts if text.strip()]
        fallback = " ".join(updates)
        if len(updates) < 2:
            return fallback
        try:
            async with asyncio.timeout(timeout_s):
                response = await llm.chat(
                    (
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(
                            role="user",
                            content="Pending spoken updates:\n" + json.dumps(updates, ensure_ascii=False),
                        ),
                    ),
                    max_tokens=max_tokens,
                    temperature=0.0,
                    enable_thinking=False,
                    timeout=timeout_s,
                )
        except (ModelServiceError, TimeoutError):
            logger.exception("voice update combiner failed")
            return fallback
        text = response.content.strip()
        if not text or response.tool_calls:
            logger.warning("voice update combiner returned no usable text")
            return fallback
        if response.finish_reason in {"length", "max_tokens"}:
            # A truncated rewrite would silently drop facts from the updates.
            logger.warning("voice update combiner rewrite hit the token limit")
            return fallback
        return text

    return combine
