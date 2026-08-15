# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice notifications derived from instrument-monitoring runtime events."""

from __future__ import annotations

from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceOutput

from .events import (
    INSTRUMENT_CHANGE_TOPIC,
    INSTRUMENT_LOST_TOPIC,
    InstrumentChange,
    InstrumentLost,
)


class InstrumentAlertAgent(Agent):
    """Turn only actionable instrument events into participant voice notes."""

    def __init__(self) -> None:
        super().__init__()

    @subscribe(INSTRUMENT_CHANGE_TOPIC)
    async def reading_changed(
        self,
        event: InstrumentChange,
        ctx: RuntimeContext,
    ) -> None:
        if event.change_type == "discovered":
            text = f"Now tracking {event.qr_text} at {event.meter_reading}."
        else:
            text = (
                f"{event.qr_text} changed from {event.previous_reading} "
                f"to {event.meter_reading}."
            )
        await ctx.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text=text, timestamp_us=event.timestamp_us),
        )

    @subscribe(INSTRUMENT_LOST_TOPIC)
    async def instrument_lost(
        self,
        event: InstrumentLost,
        ctx: RuntimeContext,
    ) -> None:
        await ctx.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(
                text=(
                    f"I am no longer tracking {event.qr_text}. "
                    f"Its last reading was {event.meter_reading}."
                ),
                timestamp_us=event.timestamp_us,
            ),
        )


__all__ = ["InstrumentAlertAgent"]
