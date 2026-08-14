# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record accepted voice and typed queries through the file-output agent."""

from __future__ import annotations

from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import UserQuery

from .events import TRANSCRIPT_RECORD_TOPIC, USER_QUERY_TOPIC, TranscriptRecord


class TranscriptAgent(Agent):
    """Convert accepted user turns into participant-scoped transcript records."""

    @subscribe(USER_QUERY_TOPIC)
    async def record(self, query: UserQuery, ctx: RuntimeContext) -> None:
        await ctx.publish(
            TRANSCRIPT_RECORD_TOPIC,
            TranscriptRecord(timestamp_us=query.timestamp_us, text=query.text),
        )


__all__ = ["TranscriptAgent"]
