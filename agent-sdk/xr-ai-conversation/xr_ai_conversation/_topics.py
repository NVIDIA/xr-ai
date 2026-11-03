# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal runtime topic bridge used only within ``xr_ai_conversation``."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe

from ._context_update import ConversationContextUpdate

if TYPE_CHECKING:
    from ._manager import ContextManager

_logger = logging.getLogger(__name__)


class _ConversationContextAgent(Agent):
    def __init__(self, manager: ContextManager) -> None:
        super().__init__()
        self._manager = manager


def _handler(source: str, mode: Literal["summary", "incremental"]) -> Any:
    async def receive(self, update: ConversationContextUpdate, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            _logger.error("dropping context update without participant source=%r", source)
            return
        await self._manager.update(participant_id, source, update, mode=mode, timestamp_us=ctx.metadata.timestamp_us)

    return receive


def make_context_subscriber(
    manager: ContextManager,
    summary_topics: Sequence[Topic[ConversationContextUpdate]],
    incremental_topics: Sequence[Topic[ConversationContextUpdate]],
) -> Agent:
    """Bind configured named topics to one context manager."""

    if not summary_topics and not incremental_topics:
        raise ValueError("at least one context topic is required")
    configured_topics = (
        *((topic, "summary") for topic in summary_topics),
        *((topic, "incremental") for topic in incremental_topics),
    )
    names = [topic.name for topic, _mode in configured_topics]
    if len(set(names)) != len(names):
        raise ValueError("context topic names must be unique")
    namespace: dict[str, Any] = {"__module__": __name__}
    for index, (topic, mode) in enumerate(configured_topics):
        namespace[f"receive_context_{index}"] = subscribe(topic)(_handler(topic.name, mode))
    subscriber_type = type("ConversationContextTopicAgent", (_ConversationContextAgent,), namespace)
    return subscriber_type(manager)


__all__ = ["make_context_subscriber"]
