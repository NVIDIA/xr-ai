# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed-data ingress for participant-aware assistants."""
from __future__ import annotations

from collections.abc import Callable, Iterable

from loguru import logger
from xr_ai_hub import DataMessage

from ._session import VoiceSession

QueryTransform = Callable[[str], str]


class TextMessageInput:
    """Route hub data messages through an assistant's normal query path."""

    def __init__(
        self,
        *,
        session: VoiceSession,
        ignore_topics: Iterable[str] = (),
        transform: QueryTransform | None = None,
        fresh_match: bool = False,
    ) -> None:
        self._session = session
        self._ignore_topics = frozenset(ignore_topics)
        self._transform = transform or (lambda text: text)
        self._fresh_match = fresh_match
        session.transport.endpoint.on_data(self._on_data)

    async def _on_data(self, message: DataMessage) -> None:
        if message.topic in self._ignore_topics:
            return
        text = (message.data or b"").decode("utf-8", errors="replace").strip()
        if not text or not self._session.is_running:
            return
        if not self._session.transport.target_participant:
            self._session.transport.set_target_participant(message.participant_id)
        text = self._transform(text)
        logger.info("text input pid={!r} {!r}", message.participant_id, text[:80])
        await self._session.enqueue_query(
            message.participant_id,
            text,
            fresh_match=self._fresh_match,
            pts_us=message.pts_us,
        )
