# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime topic bridge for named conversation context sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe

from .contracts import ConversationContextUpdate

if TYPE_CHECKING:
    from .conversation import Conversation


class _ConversationContextAgent(Agent):
    def __init__(self, conversation: Conversation) -> None:
        super().__init__()
        self._conversation = conversation


def _handler(source: str) -> Any:
    async def receive(self, update: ConversationContextUpdate, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self._conversation._update_context(None, source, update)
            return
        await self._conversation._update_context(participant_id, source, update)

    return receive


def make_context_subscriber(
    conversation: Conversation,
    topics: Sequence[Topic[ConversationContextUpdate]],
) -> Agent:
    """Bind configured named topics to one conversation context store."""

    if not topics:
        raise ValueError("at least one context topic is required")
    names = [topic.name for topic in topics]
    if len(set(names)) != len(names):
        raise ValueError("context topic names must be unique")
    namespace: dict[str, Any] = {"__module__": __name__}
    for index, topic in enumerate(topics):
        namespace[f"receive_context_{index}"] = subscribe(topic)(_handler(topic.name))
    subscriber_type = type("ConversationContextTopicAgent", (_ConversationContextAgent,), namespace)
    return subscriber_type(conversation)


__all__ = ["make_context_subscriber"]
