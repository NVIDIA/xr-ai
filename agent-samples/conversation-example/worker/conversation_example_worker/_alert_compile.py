# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile a participant's alert request into a standalone visual watch."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_routing import AgentExchange

_COMPILE = (Path(__file__).parent / "prompts" / "alert_compile.txt").read_text(encoding="utf-8").strip()


@dataclass(frozen=True, slots=True)
class CompiledAlert:
    """One watch resolved from a request and the participant's context."""

    condition: str
    """Visible condition, phrased as a clause that completes "alert me when ..."."""

    relevance: str
    """Objects or scene elements the condition is about; screens frames before the condition is judged."""


async def compile_alert(
    llm: LLMService,
    request: str,
    *,
    history: Sequence[AgentExchange] = (),
    context: str = "",
) -> CompiledAlert | None:
    """Resolve ``request`` against recent exchanges and saved context.

    :returns: The watch, or ``None`` when no specific visible condition can be
        determined, including when the model's answer is malformed.
    """

    sections = [f"Alert request:\n{request}"]
    if history:
        exchanges = "\n".join(f"User: {exchange.user}\nAssistant: {exchange.assistant}" for exchange in history)
        sections.append(f"Recent exchanges, oldest first:\n{exchanges}")
    if context:
        sections.append(f"Saved context:\n{context}")
    response = await llm.chat(
        (
            ChatMessage(role="system", content=_COMPILE),
            ChatMessage(role="user", content="\n\n".join(sections)),
        )
    )
    return _parse(response.content)


def _parse(text: str) -> CompiledAlert | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("alert compilation returned malformed output")
        return None
    if not isinstance(data, dict) or data.get("status") != "ready":
        return None
    condition, relevance = data.get("condition"), data.get("relevance")
    if not all(isinstance(value, str) and value.strip() for value in (condition, relevance)):
        logger.warning("alert compilation returned an incomplete watch")
        return None
    return CompiledAlert(condition.strip(), relevance.strip())
