# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice delivery for important visual change-watch facts."""

from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VOICE_CONTRIBUTION_TOPIC, VoiceOutput

from .events import BACKGROUND_FACT_TOPIC, BackgroundFact


class _ChangeWatchVoiceAgent(Agent):
    """Speak important facts produced by visual change watching."""

    def __init__(self) -> None:
        super().__init__()

    @subscribe(BACKGROUND_FACT_TOPIC)
    async def notify(self, fact: BackgroundFact, ctx: RuntimeContext) -> None:
        if fact.application != "change_watch":
            return
        await ctx.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text=fact.text, timestamp_us=fact.timestamp_us),
        )
