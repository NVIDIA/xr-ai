# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warn about sustained loss of usable media during a procedure capture.

Frame metadata measures arrival quality independently of snapshot sampling and
caption inference. Client telemetry supplies camera intent and network evidence;
missing telemetry is unknown, never proof of a broken stream. Each connection
owns a monitor, and each capture owns its warning task and voice-output source.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from xr_ai_hub import DataMessage, FrameSignal
from xr_ai_interaction import ResponseOutputs
from xr_ai_voice import VOICE_OUTPUT_TOPIC

CONTROL_TOPIC = "procedure.capture.quality.control"
TELEMETRY_TOPIC = "procedure.capture.quality.telemetry"
_MESSAGES = yaml.safe_load((Path(__file__).resolve().parents[3] / "yaml/capture-messages.yaml").read_text())


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = Field(default=True, strict=True)
    window_s: float = Field(default=5, gt=0)
    stall_s: float = Field(default=3, gt=0)
    repeat_interval_s: float = Field(default=10, gt=0)
    poll_interval_s: float = Field(default=1, gt=0)
    setup_duration_s: float = Field(default=10, gt=0)
    telemetry_ttl_s: float = Field(default=4, gt=0)
    min_short_edge: int = Field(default=480, gt=0, strict=True)
    drop_ratio: float = Field(default=0.5, gt=0, lt=1)


class Telemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    version: int = Field(ge=2, le=2)
    token: str = Field(min_length=1, max_length=64)
    sequence: int = Field(ge=0)
    mode: str = Field(pattern="^(off|live|on-demand)$")
    profile: str = Field(max_length=256)
    track_id: str = Field(max_length=256)
    streaming: bool
    quality: str = Field(pattern="^(unknown|excellent|good|poor|lost)$")
    rtt_ms: float | None = Field(default=None, ge=0)
    receive_jitter_ms: float | None = Field(default=None, ge=0)
    uplink_loss_fraction: float | None = Field(default=None, ge=0, le=1)
    uplink_rtt_ms: float | None = Field(default=None, ge=0)
    sent_fps: float | None = Field(default=None, ge=0)
    sent_width: int | None = Field(default=None, gt=0)
    sent_height: int | None = Field(default=None, gt=0)
    limitation: str | None = Field(default=None, pattern="^(none|bandwidth|cpu|other)$")


class QualityState:
    """Evaluate transport and resolution using local monotonic receipt times."""

    def __init__(self, config: QualityConfig, capture_fps: float, *, clock: Callable[[], float] = time.monotonic):
        self.config, self.capture_fps, self.clock = config, capture_fps, clock
        self.telemetry: Telemetry | None = None
        self.telemetry_at = float("-inf")
        self.frames: deque[float] = deque(maxlen=4096)
        self.samples: deque[float] = deque(maxlen=4096)
        self.profile = None
        self.started = clock()
        self.last_frame = None
        self.last_sequence = -1
        self.resolution = None
        self.resolution_at = float("-inf")
        self.baseline_fps = None
        self.baseline_pixels = None
        self.baseline_rates: deque[float] = deque(maxlen=3)
        self.image_failures = 0
        self.failure_at = float("-inf")
        self.active: set[str] = set()
        self.metrics: dict = {}

    def telemetry_received(self, value: Telemetry):
        now = self.clock()
        profile = (value.mode, value.profile, value.track_id, value.streaming)
        if profile != self.profile:
            self.profile, self.started = profile, now
            self.frames.clear()
            self.last_frame, self.resolution = None, None
            self.last_sequence = -1
            self.baseline_fps = self.baseline_pixels = None
            self.baseline_rates.clear()
            self.image_failures = 0
            self.active.clear()
        self.telemetry, self.telemetry_at = value, now

    def frame(self, frame: FrameSignal):
        data = self.telemetry
        if (
            data is None
            or data.mode != "live"
            or not data.streaming
            or not data.track_id
            or frame.track_id != data.track_id
            or frame.seq <= self.last_sequence
        ):
            return
        now = self.clock()
        self.frames.append(now)
        self.last_frame, self.last_sequence = now, frame.seq
        self.resolution, self.resolution_at = (frame.width, frame.height), now

    def sample(self, frame):
        now = self.clock()
        if frame is None:
            self.image_failures += 1
            self.failure_at = now
        else:
            self.samples.append(now)
            self.image_failures = 0
            self.resolution, self.resolution_at = (frame.width, frame.height), now

    def evaluate(self) -> set[str]:
        now, c, data = self.clock(), self.config, self.telemetry
        while self.frames and self.frames[0] < now - c.window_s:
            self.frames.popleft()
        while self.samples and self.samples[0] < now - c.window_s:
            self.samples.popleft()
        observed: dict[str, bool | None] = dict.fromkeys(name for name in _MESSAGES["conditions"])
        fresh = data is not None and now - self.telemetry_at <= c.telemetry_ttl_s
        fps = len(self.frames) / c.window_s
        if fresh:
            observed["poor_network"] = data.quality in {"poor", "lost"} if data.quality != "unknown" else None
            live = data.mode == "live" and data.streaming and bool(data.track_id)
            if live:
                age = now - (self.last_frame if self.last_frame is not None else self.started)
                observed["video_stalled"] = age >= c.stall_s
                if now - self.started >= c.window_s and age < c.stall_s:
                    if self.baseline_fps is None and fps > 0:
                        self.baseline_rates.append(fps)
                        if len(self.baseline_rates) == 3 and min(self.baseline_rates) >= 0.8 * max(self.baseline_rates):
                            self.baseline_fps = sum(self.baseline_rates) / 3
                    observed["low_frame_rate"] = fps < self.capture_fps or (
                        self.baseline_fps is not None and fps < self.baseline_fps * c.drop_ratio
                    )
        if (
            self.resolution is not None
            and now - self.resolution_at <= 2 * c.window_s
            and not (fresh and data.mode == "off")
        ):
            width, height = self.resolution
            pixels = width * height
            if self.baseline_pixels is None:
                self.baseline_pixels = pixels
            observed["low_resolution"] = (
                min(width, height) < c.min_short_edge or pixels < self.baseline_pixels * c.drop_ratio
            )
        # On-demand capture has no incoming video stream. Only failed, actually
        # requested snapshots establish a stall; missing telemetry alone does not.
        if fresh and data.mode == "on-demand" and self.image_failures:
            last = self.samples[-1] if self.samples else self.started
            observed["video_stalled"] = now - last >= c.stall_s
        self.active = {name for name, bad in observed.items() if bad is True}
        self.metrics = {
            "telemetry_available": fresh,
            "saved_fps": len(self.samples) / c.window_s,
            "received_fps": fps if fresh and data.mode == "live" else None,
            "resolution": self.resolution,
            "baseline_fps": self.baseline_fps,
            "client": data.model_dump(exclude={"token"}) if fresh else None,
        }
        # A stall subsumes a rate warning and avoids two messages for one outage.
        return self.active - ({"low_frame_rate"} if "video_stalled" in self.active else set())


class QualityFeedback:
    """Own one connection's telemetry lease and each capture's warning lifetime."""

    def __init__(
        self,
        pid,
        endpoint,
        backend,
        config: QualityConfig,
        capture_fps: float,
        *,
        clock=time.monotonic,
        sleep=asyncio.sleep,
    ):
        self.pid, self.endpoint, self.backend = pid, endpoint, backend
        self.clock, self.sleep = clock, sleep
        self.config, self.capture_fps = config, capture_fps
        self.token = uuid4().hex
        self.sequence = -1
        self.control_sequence = 0
        self.latest = None
        self.latest_at = float("-inf")
        self.state: QualityState | None = None
        self.task: asyncio.Task | None = None
        self.active = False

    async def connect(self):
        await self._control()

    async def _control(self):
        self.control_sequence += 1
        message = {"version": 1, "token": self.token, "sequence": self.control_sequence, "active": self.active}
        try:
            async with asyncio.timeout(2):
                await self.endpoint.send_return_data(
                    DataMessage(
                        self.pid,
                        CONTROL_TOPIC,
                        time.time_ns() // 1000,
                        json.dumps(message).encode(),
                    )
                )
        except Exception as exc:  # noqa: BLE001 — feedback cannot block capture/finalization on transport failure.
            logger.warning("capture quality control unavailable pid={!r}: {}", self.pid, exc)

    def telemetry(self, msg: DataMessage):
        if msg.topic != TELEMETRY_TOPIC or len(msg.data) > 4096:
            return
        try:
            data = Telemetry.model_validate_json(msg.data)
        except ValidationError:
            return
        if data.token != self.token or data.sequence <= self.sequence:
            return
        self.sequence, self.latest, self.latest_at = data.sequence, data, self.clock()
        if self.state is not None:
            self.state.telemetry_received(data)

    def frame(self, frame):
        if self.state is not None:
            self.state.frame(frame)

    def sample(self, frame):
        if self.state is not None:
            self.state.sample(frame)

    def start(self, publisher):
        if self.task is not None and not self.task.done():
            raise RuntimeError("previous capture quality task must be drained before restart")
        self.state = QualityState(self.config, self.capture_fps, clock=self.clock)
        if self.clock() - self.latest_at <= self.config.telemetry_ttl_s:
            self.state.telemetry_received(self.latest)
        self.active = True
        self.task = asyncio.create_task(self._run(publisher), name=f"capture-quality:{self.pid}")

    def invalidate(self):
        self.active = False
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()

    async def stop(self):
        self.invalidate()
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        self.state = None
        await self._control()

    async def _record(self, kind, **data):
        await self.backend.capture_quality(self.pid, {"kind": kind, **data})

    async def _run(self, publisher):
        outputs = ResponseOutputs(publisher, VOICE_OUTPUT_TOPIC, source=f"capture-quality:{uuid4().hex}")
        schedule = NotificationSchedule(self.config.repeat_interval_s)
        previous = set()
        heartbeat = float("-inf")
        try:
            await self._record("monitor_started", policy=self.config.model_dump())
            while self.active:
                now = self.clock()
                conditions = self.state.evaluate()
                if conditions != previous:
                    # Withdraw a queued description of conditions that changed.
                    await outputs.withdraw(self.pid)
                    await self._record("conditions_changed", conditions=sorted(conditions), metrics=self.state.metrics)
                    previous = set(conditions)
                due = schedule.due(conditions, now)
                if due:
                    text = " ".join(_MESSAGES["conditions"][name] for name in due)
                    await outputs.update(self.pid, text, urgent=True)
                    schedule.sent(due, self.clock())
                    await self._record("warning_submitted", conditions=due, text=text)
                if now - heartbeat >= 2:
                    await self._control()
                    heartbeat = now
                await self.sleep(self.config.poll_interval_s)
        except Exception as exc:  # noqa: BLE001 — feedback failure must never stop capture.
            logger.exception("capture quality feedback failed pid={!r}: {}", self.pid, exc)
        finally:
            try:
                await outputs.withdraw(self.pid)
                await self._record("monitor_stopped")
            except Exception as exc:  # noqa: BLE001 — voice can close before diagnostics cleanup.
                logger.warning("capture quality cleanup unavailable pid={!r}: {}", self.pid, exc)


class NotificationSchedule:
    """Coalesce conditions due on this tick, with an independent repeat clock each.

    Recovery does not erase the last notification time: even a flapping condition
    cannot announce more often than repeat_interval_s. A different condition is
    immediately eligible. Only successfully submitted messages advance clocks.
    """

    def __init__(self, repeat_interval_s: float):
        self.interval = repeat_interval_s
        self.last_sent: dict[str, float] = {}

    def due(self, conditions: set[str], now: float) -> list[str]:
        return [
            name
            for name in _MESSAGES["conditions"]
            if name in conditions and now - self.last_sent.get(name, float("-inf")) >= self.interval
        ]

    def sent(self, conditions, now: float) -> None:
        self.last_sent.update(dict.fromkeys(conditions, now))
