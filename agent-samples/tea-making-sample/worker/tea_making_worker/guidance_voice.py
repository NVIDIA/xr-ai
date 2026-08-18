# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice delivery for actionable tea-guidance notices."""

from __future__ import annotations

from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceOutput

from .events import GUIDANCE_NOTICE_TOPIC, GuidanceNotice


class GuidanceVoiceAgent(Agent):
    """Speak workflow notices while background applications remain file-only."""

    def __init__(self) -> None:
        super().__init__()

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def notify(self, notice: GuidanceNotice, ctx: RuntimeContext) -> None:
        await ctx.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text=notice.text, timestamp_us=notice.timestamp_us),
        )


__all__ = ["GuidanceVoiceAgent"]
