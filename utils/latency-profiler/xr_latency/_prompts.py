# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build prompts of a target token size and calibrate chars-per-token against the live server."""

from __future__ import annotations

import secrets

from xr_ai_models import ChatMessage, LLMService

_PARAGRAPH = (
    "The maintenance technician walked along the east wall of the workshop, checking each instrument panel in turn. "
    "The first gauge read normal pressure, the second showed a slow drift in temperature, and the third had a loose "
    "label that needed to be reattached before the afternoon shift began. She noted each reading in the logbook, "
    "photographed the drifting gauge for the record, and moved on to the compressor housing at the far corner. "
)


class PromptBuilder:
    """Produces user prompts whose server-side token count is close to a requested size."""

    def __init__(self, tokens_per_char: float = 0.22) -> None:
        self.tokens_per_char = tokens_per_char

    async def calibrate(self, llm: LLMService) -> float:
        """Measure tokens per character on the live tokenizer using reported usage, and keep it."""
        text = _PARAGRAPH * 8
        response = await llm.chat(
            [ChatMessage(role="user", content=text)], max_tokens=1, enable_thinking=False, temperature=0.0,
        )
        used = response.raw["usage"]["prompt_tokens"]
        # Subtract the chat-template overhead by differencing against a tiny prompt.
        tiny = await llm.chat(
            [ChatMessage(role="user", content="hi")], max_tokens=1, enable_thinking=False, temperature=0.0,
        )
        overhead = tiny.raw["usage"]["prompt_tokens"] - 1
        self.tokens_per_char = max(used - overhead, 1) / len(text)
        return self.tokens_per_char

    def filler(self, tokens: int, *, nonce: bool) -> str:
        """Return ~``tokens`` tokens of text; ``nonce`` puts a unique prefix first to defeat prefix caching."""
        chars = max(int(tokens / self.tokens_per_char), 1)
        body = (_PARAGRAPH * (chars // len(_PARAGRAPH) + 1))[:chars]
        return f"[{secrets.token_hex(8)}] {body}" if nonce else body

    def user_message(self, tokens: int, question: str, *, nonce: bool) -> ChatMessage:
        """A user message made of ``tokens`` of context followed by ``question``."""
        return ChatMessage(role="user", content=f"{self.filler(tokens, nonce=nonce)}\n\n{question}")
