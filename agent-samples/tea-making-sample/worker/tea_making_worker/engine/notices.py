# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feed background step notices through the normal voice output path."""

from __future__ import annotations

import uuid

from xr_ai_voice import VoiceSession

from ..runtime.events import emit


class NoticeBridge:
    def __init__(self, session: VoiceSession) -> None:
        self._session = session
        self._pending: dict[str, str] = {}

    async def send(self, participant_id: str, message: str) -> None:
        if not self._session.is_running:
            emit("notice.dropped", participant_id=participant_id, message=message)
            return
        token = f"__guide_notice_{uuid.uuid4().hex}"
        self._pending[token] = message
        emit("notice.queued", participant_id=participant_id, message=message)
        await self._session.enqueue_query(participant_id, token, fresh_match=True)

    def take(self, token: str) -> str | None:
        return self._pending.pop(token, None)


__all__ = ["NoticeBridge"]
