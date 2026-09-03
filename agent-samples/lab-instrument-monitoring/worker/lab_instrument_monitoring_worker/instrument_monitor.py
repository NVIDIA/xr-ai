# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful monitoring for readings produced by the lab instrument reader."""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import nemo_relay
from loguru import logger
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools import Tool
from xr_ai_tools.marker_tracking import MarkerType
from xr_ai_voice import VoiceParticipantLeft

from .events import (
    _INSTRUMENT_TRACKING_TOPIC,
    INSTRUMENT_CHANGE_TOPIC,
    INSTRUMENT_LOST_TOPIC,
    INSTRUMENT_STATE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    InstrumentChange,
    InstrumentLost,
    InstrumentReading,
    InstrumentSighting,
    InstrumentState,
    InstrumentStateSnapshot,
    _InstrumentTrackingUpdate,
)
from .instruments import LabInstrumentAgent, ReadLabInstrumentsRequest
from .monitor import MonitoringRequest, MonitoringState

_NUMBER = re.compile(r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?")
_UNIT = re.compile(r"(?:°\s*)?[A-Za-zµμΩ%]+(?:\s*/\s*[A-Za-z]+)?")
_UNIT_ALIASES = {
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "celsius": "°C",
    "degree c": "°C",
    "degrees c": "°C",
    "fahrenheit": "°F",
    "degree f": "°F",
    "degrees f": "°F",
    "hertz": "Hz",
    "ohm": "Ω",
    "ohms": "Ω",
    "percent": "%",
    "volt": "V",
    "volts": "V",
    "watt": "W",
    "watts": "W",
}
_TRACKING_SCAN_INTERVAL_S = 0.5


@dataclass(frozen=True, slots=True)
class _ReadingValue:
    value: Decimal
    unit: str

    @property
    def display(self) -> str:
        number = format(self.value, "f")
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        return f"{number} {self.unit}".strip()


@dataclass(slots=True)
class _TrackedInstrument:
    value: Decimal
    unit: str
    state: InstrumentState
    last_seen_monotonic: float


@dataclass(slots=True)
class _TrackedMarkerPresence:
    marker_type: MarkerType
    marker_id: str
    device_name: str
    first_seen_us: int
    last_seen_us: int
    last_seen_monotonic: float
    tracking: bool = True


@dataclass(slots=True)
class _ParticipantTracker:
    instruments: dict[tuple[MarkerType, str], _TrackedInstrument] = field(default_factory=dict)
    markers: dict[tuple[MarkerType, str], _TrackedMarkerPresence] = field(
        default_factory=dict
    )
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def normalize_meter_reading(
    reading: str,
    *,
    previous_unit: str = "",
) -> tuple[Decimal, str, str] | None:
    """Extract a stable numeric identity and retain a previously known unit."""

    match = _NUMBER.search(reading)
    if match is None:
        return None
    token = match.group(0)
    if "," in token and "." in token:
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None

    suffix = reading[match.end() :].strip(" :=-_")
    unit_match = _UNIT.match(suffix)
    unit = previous_unit
    if unit_match is not None:
        raw_unit = re.sub(r"\s+", " ", unit_match.group(0)).strip()
        compact = raw_unit.replace(" ", "")
        unit = _UNIT_ALIASES.get(raw_unit.casefold(), compact)
    parsed = _ReadingValue(value=value, unit=unit)
    return parsed.value, parsed.unit, parsed.display


class InstrumentMonitorAgent(Agent):
    """Track participant-scoped instrument state and publish meaningful events."""

    def __init__(
        self,
        *,
        reader: LabInstrumentAgent,
        interval_s: float,
        snapshot_interval_s: float = 10.0,
        lost_after_s: float = 15.0,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if snapshot_interval_s <= 0:
            raise ValueError("snapshot_interval_s must be positive")
        if lost_after_s <= 0:
            raise ValueError("lost_after_s must be positive")
        self._reader = reader
        self._interval_s = interval_s
        self._snapshot_interval_s = snapshot_interval_s
        self._lost_after_s = lost_after_s
        self._runtime: AgentRuntime | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._trackers: dict[str, _ParticipantTracker] = {}
        self._stopped = False
        self.start_instrument_monitoring = Tool(
            "start_instrument_monitoring",
            "Continuously track marker-labelled lab instrument readings.",
            MonitoringRequest,
            MonitoringState,
            self._start_monitoring,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        self.stop_instrument_monitoring = Tool(
            "stop_instrument_monitoring",
            "Stop tracking marker-labelled lab instrument readings.",
            MonitoringRequest,
            MonitoringState,
            self._stop_monitoring,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        self.instrument_monitoring_status = Tool(
            "instrument_monitoring_status",
            "Report whether marker-labelled lab instrument tracking is active.",
            MonitoringRequest,
            MonitoringState,
            self._monitoring_status,
            return_direct=True,
            render_result=lambda result: result.message,
        )
        super().__init__(
            (
                self.start_instrument_monitoring,
                self.stop_instrument_monitoring,
                self.instrument_monitoring_status,
            )
        )

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if not runtime.running:
            raise RuntimeError("instrument monitor requires a running agent runtime")
        if self._runtime is not None:
            raise RuntimeError("instrument monitor is already started")
        self._runtime = runtime

    async def _start_monitoring(self, request: MonitoringRequest) -> MonitoringState:
        if self._stopped:
            raise RuntimeError("instrument monitor is stopping")
        if self._runtime is None or not self._runtime.running:
            raise RuntimeError("instrument monitor requires a running agent runtime")
        task = self._tasks.get(request.participant_id)
        if task is not None and not task.done():
            return MonitoringState(active=True, message="Lab instrument monitoring is already running.")
        self._trackers[request.participant_id] = _ParticipantTracker()
        task = asyncio.create_task(
            self._run(request.participant_id),
            name=f"instrument-monitor:{request.participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[request.participant_id] = task
        task.add_done_callback(lambda completed, pid=request.participant_id: self._discard(pid, completed))
        return MonitoringState(active=True, message="Lab instrument monitoring started.")

    async def _stop_monitoring(self, request: MonitoringRequest) -> MonitoringState:
        active = await self._cancel(request.participant_id)
        return MonitoringState(
            active=False,
            message=("Lab instrument monitoring stopped." if active else "Lab instrument monitoring is not running."),
        )

    async def _monitoring_status(self, request: MonitoringRequest) -> MonitoringState:
        task = self._tasks.get(request.participant_id)
        active = task is not None and not task.done()
        return MonitoringState(
            active=active,
            message=("Lab instrument monitoring is running." if active else "Lab instrument monitoring is stopped."),
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(self, _event: VoiceParticipantLeft, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            await self._cancel(participant_id)

    async def _run(self, participant_id: str) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._tracking_scan_loop(participant_id))
                tasks.create_task(self._scan_loop(participant_id))
                tasks.create_task(self._maintenance_loop(participant_id))

    async def _tracking_scan_loop(self, participant_id: str) -> None:
        while True:
            sightings = await self._reader._scan_instrument_sightings(participant_id)
            await self._observe_tracking(participant_id, sightings)
            await asyncio.sleep(_TRACKING_SCAN_INTERVAL_S)

    async def _scan_loop(self, participant_id: str) -> None:
        next_scan = time.monotonic()
        while True:
            try:
                result = await self._reader.read_lab_instruments.execute(
                    ReadLabInstrumentsRequest(participant_id=participant_id)
                )
                await self._observe(
                    participant_id,
                    result.readings,
                    sightings=result.sightings,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning("instrument scan failed pid={!r}", participant_id)
            next_scan += self._interval_s
            now = time.monotonic()
            if now >= next_scan:
                missed = int((now - next_scan) // self._interval_s) + 1
                next_scan += missed * self._interval_s
            await asyncio.sleep(next_scan - now)

    async def _maintenance_loop(self, participant_id: str) -> None:
        next_snapshot = time.monotonic() + self._snapshot_interval_s
        tick_s = min(1.0, self._snapshot_interval_s, self._lost_after_s)
        while True:
            await asyncio.sleep(tick_s)
            now = time.monotonic()
            await self._publish_lost(participant_id, now)
            if now >= next_snapshot:
                await self._publish_snapshot(participant_id)
                missed = int((now - next_snapshot) // self._snapshot_interval_s)
                next_snapshot += (missed + 1) * self._snapshot_interval_s

    async def _observe(
        self,
        participant_id: str,
        readings: list[InstrumentReading],
        *,
        sightings: list[InstrumentSighting] | None = None,
        observed_at: float | None = None,
    ) -> None:
        tracking_sightings = sightings
        if tracking_sightings is None:
            tracking_sightings = [
                InstrumentSighting(
                    timestamp_us=reading.timestamp_us,
                    marker_type=reading.marker_type,
                    marker_id=reading.marker_id,
                    device_name=reading.device_name,
                )
                for reading in readings
            ]
        await self._observe_tracking(
            participant_id,
            tracking_sightings,
            observed_at=observed_at,
        )
        tracker = self._trackers.get(participant_id)
        runtime = self._runtime
        if tracker is None or runtime is None:
            return
        seen_at = time.monotonic() if observed_at is None else observed_at
        changes: list[InstrumentChange] = []
        async with tracker.lock:
            unique_sightings = {
                (item.marker_type, item.marker_id): item
                for item in sightings or ()
            }
            for marker_key, sighting in unique_sightings.items():
                previous = tracker.instruments.get(marker_key)
                if previous is None:
                    continue
                previous.last_seen_monotonic = seen_at
                previous.state.last_seen_us = sighting.timestamp_us
                previous.state.tracking = True

            unique = {(item.marker_type, item.marker_id): item for item in readings}
            for marker_key, reading in unique.items():
                previous = tracker.instruments.get(marker_key)
                normalized = normalize_meter_reading(
                    reading.meter_reading,
                    previous_unit=previous.unit if previous is not None else "",
                )
                if normalized is None:
                    logger.warning(
                        "ignoring unstructured instrument reading pid={!r} marker={!r} reading={!r}",
                        participant_id,
                        marker_key,
                        reading.meter_reading,
                    )
                    continue
                value, unit, display = normalized
                if previous is None:
                    state = InstrumentState(
                        marker_type=reading.marker_type,
                        marker_id=reading.marker_id,
                        device_name=reading.device_name,
                        meter_reading=display,
                        first_seen_us=reading.timestamp_us,
                        last_seen_us=reading.timestamp_us,
                    )
                    tracker.instruments[marker_key] = _TrackedInstrument(
                        value=value,
                        unit=unit,
                        state=state,
                        last_seen_monotonic=seen_at,
                    )
                    changes.append(
                        InstrumentChange(
                            timestamp_us=reading.timestamp_us,
                            change_type="discovered",
                            marker_type=reading.marker_type,
                            marker_id=reading.marker_id,
                            device_name=reading.device_name,
                            meter_reading=display,
                            last_seen_us=reading.timestamp_us,
                        )
                    )
                    continue

                old_display = previous.state.meter_reading
                previous.last_seen_monotonic = seen_at
                previous.state.last_seen_us = reading.timestamp_us
                previous.state.tracking = True
                previous.state.meter_reading = display
                if (value, unit) == (previous.value, previous.unit):
                    continue
                previous.value = value
                previous.unit = unit
                changes.append(
                    InstrumentChange(
                        timestamp_us=reading.timestamp_us,
                        change_type="reading_changed",
                        marker_type=reading.marker_type,
                        marker_id=reading.marker_id,
                        device_name=reading.device_name,
                        previous_reading=old_display,
                        meter_reading=display,
                        last_seen_us=reading.timestamp_us,
                    )
                )
        for change in changes:
            try:
                await runtime.publish(
                    INSTRUMENT_CHANGE_TOPIC,
                    change,
                    participant_id=participant_id,
                    source="instrument-monitor",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning(
                    "instrument change publish failed pid={!r} device={!r}",
                    participant_id,
                    change.device_name,
                )

    async def _observe_tracking(
        self,
        participant_id: str,
        sightings: list[InstrumentSighting],
        *,
        observed_at: float | None = None,
    ) -> None:
        tracker = self._trackers.get(participant_id)
        runtime = self._runtime
        if tracker is None or runtime is None:
            return
        seen_at = time.monotonic() if observed_at is None else observed_at
        updates: list[_InstrumentTrackingUpdate] = []
        async with tracker.lock:
            unique = {
                (sighting.marker_type, sighting.marker_id): sighting
                for sighting in sightings
            }
            for marker_key, sighting in unique.items():
                presence = tracker.markers.get(marker_key)
                reading = tracker.instruments.get(marker_key)
                if presence is None:
                    presence = _TrackedMarkerPresence(
                        marker_type=sighting.marker_type,
                        marker_id=sighting.marker_id,
                        device_name=sighting.device_name,
                        first_seen_us=sighting.timestamp_us,
                        last_seen_us=sighting.timestamp_us,
                        last_seen_monotonic=seen_at,
                    )
                    tracker.markers[marker_key] = presence
                    updates.append(
                        _InstrumentTrackingUpdate(
                            timestamp_us=sighting.timestamp_us,
                            marker_type=sighting.marker_type,
                            marker_id=sighting.marker_id,
                            device_name=sighting.device_name,
                            tracking=True,
                            meter_reading=(
                                reading.state.meter_reading if reading is not None else None
                            ),
                            last_seen_us=sighting.timestamp_us,
                        )
                    )
                else:
                    if not presence.tracking:
                        updates.append(
                            _InstrumentTrackingUpdate(
                                timestamp_us=sighting.timestamp_us,
                                marker_type=sighting.marker_type,
                                marker_id=sighting.marker_id,
                                device_name=sighting.device_name,
                                tracking=True,
                                meter_reading=(
                                    reading.state.meter_reading
                                    if reading is not None
                                    else None
                                ),
                                last_seen_us=sighting.timestamp_us,
                            )
                        )
                    presence.last_seen_us = sighting.timestamp_us
                    presence.last_seen_monotonic = seen_at
                    presence.tracking = True
                if reading is not None:
                    reading.last_seen_monotonic = seen_at
                    reading.state.last_seen_us = sighting.timestamp_us
                    reading.state.tracking = True
        await self._publish_tracking_updates(participant_id, updates)

    async def _publish_lost(self, participant_id: str, now: float) -> None:
        tracker = self._trackers.get(participant_id)
        runtime = self._runtime
        if tracker is None or runtime is None:
            return
        timestamp_us = time.time_ns() // 1_000
        lost: list[InstrumentLost] = []
        tracking_updates: list[_InstrumentTrackingUpdate] = []
        async with tracker.lock:
            for marker_key, presence in tracker.markers.items():
                if (
                    not presence.tracking
                    or now - presence.last_seen_monotonic < self._lost_after_s
                ):
                    continue
                presence.tracking = False
                reading = tracker.instruments.get(marker_key)
                tracking_updates.append(
                    _InstrumentTrackingUpdate(
                        timestamp_us=timestamp_us,
                        marker_type=presence.marker_type,
                        marker_id=presence.marker_id,
                        device_name=presence.device_name,
                        tracking=False,
                        meter_reading=(
                            reading.state.meter_reading if reading is not None else None
                        ),
                        last_seen_us=presence.last_seen_us,
                    )
                )
            for tracked in tracker.instruments.values():
                if not tracked.state.tracking or now - tracked.last_seen_monotonic < self._lost_after_s:
                    continue
                tracked.state.tracking = False
                lost.append(
                    InstrumentLost(
                        timestamp_us=timestamp_us,
                        marker_type=tracked.state.marker_type,
                        marker_id=tracked.state.marker_id,
                        device_name=tracked.state.device_name,
                        meter_reading=tracked.state.meter_reading,
                        last_seen_us=tracked.state.last_seen_us,
                    )
                )
        await self._publish_tracking_updates(participant_id, tracking_updates)
        for event in lost:
            try:
                await runtime.publish(
                    INSTRUMENT_LOST_TOPIC,
                    event,
                    participant_id=participant_id,
                    source="instrument-monitor",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning(
                    "instrument lost publish failed pid={!r} device={!r}",
                    participant_id,
                    event.device_name,
                )

    async def _publish_tracking_updates(
        self,
        participant_id: str,
        updates: list[_InstrumentTrackingUpdate],
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        for update in updates:
            try:
                await runtime.publish(
                    _INSTRUMENT_TRACKING_TOPIC,
                    update,
                    participant_id=participant_id,
                    source="instrument-tracking",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).warning(
                    "instrument tracking publish failed pid={!r} device={!r}",
                    participant_id,
                    update.device_name,
                )

    async def _publish_snapshot(self, participant_id: str) -> None:
        tracker = self._trackers.get(participant_id)
        runtime = self._runtime
        if tracker is None or runtime is None:
            return
        async with tracker.lock:
            instruments = [
                tracked.state.model_copy(deep=True)
                for tracked in sorted(
                    tracker.instruments.values(),
                    key=lambda tracked: tracked.state.device_name,
                )
            ]
        try:
            await runtime.publish(
                INSTRUMENT_STATE_TOPIC,
                InstrumentStateSnapshot(
                    timestamp_us=time.time_ns() // 1_000,
                    instruments=instruments,
                ),
                participant_id=participant_id,
                source="instrument-monitor",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "instrument state publish failed pid={!r}",
                participant_id,
            )

    async def _cancel(self, participant_id: str) -> bool:
        task = self._tasks.pop(participant_id, None)
        self._trackers.pop(participant_id, None)
        if task is None:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
            self._trackers.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error("instrument monitor stopped pid={!r}: {!r}", participant_id, error)

    async def stop(self) -> None:
        self._stopped = True
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        self._trackers.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runtime = None


__all__ = ["InstrumentMonitorAgent", "normalize_meter_reading"]
