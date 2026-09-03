# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice notifications derived from instrument-monitoring runtime events."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import nemo_relay
from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    VoiceOutput,
    VoiceParticipantLeft,
    VoicePriority,
)

from .events import (
    _INSTRUMENT_TRACKING_TOPIC,
    INSTRUMENT_CHANGE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    InstrumentChange,
    _InstrumentTrackingUpdate,
)

_VOICE_INTERVAL_S = 5.0
_SUMMARY_TIMEOUT_S = 5.0
_SUMMARY_PROMPT = """Summarize instrument changes since the last spoken update.
Use one brief, precise sentence with no preamble or filler.
Keep instrument names, final readings, units, and tracking status. Mention only actionable reversals or anomalies.
Prefer 20 words or fewer unless more words are required to preserve those facts.
Do not invent values, units, causes, or instruments."""


@dataclass(frozen=True, slots=True)
class _Alert:
    key: tuple[str, str]
    device_name: str
    meter_reading: str
    timestamp_us: int
    lost: bool
    discovered: bool
    previous_reading: str | None
    ctx: RuntimeContext


@dataclass(slots=True)
class _ParticipantAlerts:
    last_spoken_at: float | None = None
    pending: dict[tuple[str, str], list[_Alert]] = field(default_factory=dict)
    flush_task: asyncio.Task[None] | None = None


class InstrumentAlertAgent(Agent):
    """Turn only actionable instrument events into participant voice notes."""

    def __init__(self) -> None:
        super().__init__()
        self._states: dict[str, _ParticipantAlerts] = {}
        self._lock = asyncio.Lock()
        self._llm: LLMService | None = None

    def _bind_llm(self, llm: LLMService) -> None:
        self._llm = llm

    @subscribe(INSTRUMENT_CHANGE_TOPIC)
    async def reading_changed(
        self,
        event: InstrumentChange,
        ctx: RuntimeContext,
    ) -> None:
        await self._accept(
            self._participant(ctx),
            _Alert(
                key=(str(event.marker_type), event.marker_id),
                device_name=event.device_name,
                meter_reading=event.meter_reading,
                timestamp_us=event.timestamp_us,
                lost=False,
                discovered=event.change_type == "discovered",
                previous_reading=event.previous_reading,
                ctx=ctx,
            ),
            immediate_text=(
                f"{event.device_name}: {event.meter_reading}."
                if event.change_type == "discovered"
                else f"{event.device_name}: {event.previous_reading} to {event.meter_reading}."
            ),
        )

    @subscribe(_INSTRUMENT_TRACKING_TOPIC)
    async def tracking_changed(
        self,
        event: _InstrumentTrackingUpdate,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        key = (str(event.marker_type), event.marker_id)
        async with self._lock:
            state = self._states.setdefault(participant_id, _ParticipantAlerts())
            state.pending.pop(key, None)
            state.last_spoken_at = asyncio.get_running_loop().time()
        if event.tracking:
            text = f"Tracking {event.device_name}."
        elif event.meter_reading is None:
            text = f"Lost {event.device_name}."
        else:
            text = f"Lost {event.device_name}. Last reading: {event.meter_reading}."
        await self._publish(
            ctx,
            text,
            event.timestamp_us,
            priority=VoicePriority.HIGH,
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        async with self._lock:
            state = self._states.pop(participant_id, None)
        if state is not None and state.flush_task is not None:
            state.flush_task.cancel()
            await asyncio.gather(state.flush_task, return_exceptions=True)

    async def stop(self) -> None:
        async with self._lock:
            tasks = tuple(
                state.flush_task
                for state in self._states.values()
                if state.flush_task is not None
            )
            self._states.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _accept(
        self,
        participant_id: str,
        alert: _Alert,
        *,
        immediate_text: str,
    ) -> None:
        now = asyncio.get_running_loop().time()
        speak_now = False
        async with self._lock:
            state = self._states.setdefault(participant_id, _ParticipantAlerts())
            quiet = (
                state.last_spoken_at is None
                or now - state.last_spoken_at >= _VOICE_INTERVAL_S
            )
            if quiet and not state.pending:
                state.last_spoken_at = now
                speak_now = True
            else:
                state.pending.setdefault(alert.key, []).append(alert)
                if state.flush_task is None or state.flush_task.done():
                    delay_s = max(
                        0.0,
                        _VOICE_INTERVAL_S - (now - (state.last_spoken_at or now)),
                    )
                    state.flush_task = asyncio.create_task(
                        self._flush_after(participant_id, state, delay_s),
                        name=f"instrument-voice-summary:{participant_id}",
                        context=nemo_relay.fork_asyncio_context(),
                    )
        if speak_now:
            await self._publish(
                alert.ctx,
                immediate_text,
                alert.timestamp_us,
            )

    async def _flush_after(
        self,
        participant_id: str,
        state: _ParticipantAlerts,
        delay_s: float,
    ) -> None:
        await asyncio.sleep(delay_s)
        async with self._lock:
            if self._states.get(participant_id) is not state:
                return
            histories = tuple(state.pending.values())
            state.pending.clear()
            state.flush_task = None
            if not histories:
                return
            state.last_spoken_at = asyncio.get_running_loop().time()
        text = await self._summarize(histories)
        alerts = tuple(alert for history in histories for alert in history)
        latest = max(alerts, key=lambda alert: alert.timestamp_us)
        await self._publish(latest.ctx, text, latest.timestamp_us)

    async def _summarize(self, histories: tuple[list[_Alert], ...]) -> str:
        fallback = self._fallback_summary(histories)
        if self._llm is None:
            return fallback
        transitions = [
            {
                "instrument": history[-1].device_name,
                "events": [
                    {
                        "status": (
                            "lost"
                            if alert.lost
                            else "discovered" if alert.discovered else "changed"
                        ),
                        "previous_reading": alert.previous_reading,
                        "reading": alert.meter_reading,
                    }
                    for alert in history
                ],
            }
            for history in histories
        ]
        try:
            async with asyncio.timeout(_SUMMARY_TIMEOUT_S):
                response = await self._llm.chat(
                    (
                        ChatMessage(role="system", content=_SUMMARY_PROMPT),
                        ChatMessage(
                            role="user",
                            content=json.dumps(transitions, ensure_ascii=False),
                        ),
                    ),
                    max_tokens=128,
                    temperature=0.0,
                    enable_thinking=False,
                    timeout=_SUMMARY_TIMEOUT_S,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "instrument voice summary failed; using final-state fallback"
            )
            return fallback
        text = response.content.strip()
        if not text or response.tool_calls:
            logger.warning(
                "instrument voice summary returned no usable text; using fallback"
            )
            return fallback
        return text

    @staticmethod
    def _fallback_summary(histories: tuple[list[_Alert], ...]) -> str:
        updates: list[str] = []
        for history in histories:
            first = history[0]
            last = history[-1]
            if last.lost:
                updates.append(
                    f"Lost {last.device_name} at {last.meter_reading}."
                )
            elif first.previous_reading:
                updates.append(
                    f"{last.device_name}: {first.previous_reading} to {last.meter_reading}."
                )
            else:
                updates.append(f"{last.device_name}: {last.meter_reading}.")
        return " ".join(updates)

    @staticmethod
    async def _publish(
        ctx: RuntimeContext,
        text: str,
        timestamp_us: int,
        *,
        priority: VoicePriority = VoicePriority.NORMAL,
    ) -> None:
        try:
            await ctx.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(
                    text=text,
                    priority=priority,
                    timestamp_us=timestamp_us,
                ),
            )
        except RuntimeClosedError:
            return

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("instrument alerts require a participant")
        return participant_id


__all__ = ["InstrumentAlertAgent"]
