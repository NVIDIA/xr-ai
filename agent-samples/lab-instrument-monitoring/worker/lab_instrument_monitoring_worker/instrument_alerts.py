# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice notifications derived from instrument-monitoring runtime events."""

from __future__ import annotations

from xr_ai_interaction import ResponseOutputs
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC

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
            text = f"Now tracking {event.device_name} at {event.meter_reading}."
        else:
            text = f"{event.device_name} changed from {event.previous_reading} to {event.meter_reading}."
        await self._update(ctx, text, event.timestamp_us)

    @subscribe(INSTRUMENT_LOST_TOPIC)
    async def instrument_lost(
        self,
        event: InstrumentLost,
        ctx: RuntimeContext,
    ) -> None:
        await self._update(
            ctx,
            f"I am no longer tracking {event.device_name}. Its last reading was {event.meter_reading}.",
            event.timestamp_us,
        )

    @staticmethod
    async def _update(ctx: RuntimeContext, text: str, timestamp_us: int) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        await ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC).update(participant_id, text, timestamp_us=timestamp_us)


__all__ = ["InstrumentAlertAgent"]
