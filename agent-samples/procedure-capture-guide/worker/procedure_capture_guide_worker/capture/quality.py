# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report received video frame rate and resolution independently of recording.

Each participant connection owns frame observations; each capture owns its
warning task and voice-output sources. No camera or model requests are made.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameSignal
from xr_ai_interaction import ResponseOutputs
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import VOICE_OUTPUT_TOPIC

from .audio_quality import AudioQuality

_MESSAGES = yaml.safe_load((Path(__file__).resolve().parents[3] / "yaml/capture-messages.yaml").read_text())


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = Field(default=True, strict=True)
    window_s: float = Field(default=5, gt=0)
    stall_s: float = Field(default=3, gt=0)
    repeat_interval_s: float = Field(default=10, gt=0)
    poll_interval_s: float = Field(default=1, gt=0)
    frame_rate_drop_ratio: float = Field(default=0.5, gt=0, lt=1)
    resolution_drop_ratio: float = Field(default=0.5, gt=0, lt=1)
    audio_snr_drop_ratio: float = Field(default=0.5, gt=0, lt=1)
    audio_noise_window_s: float = Field(default=5, ge=1, le=30)
    audio_min_noise_s: float = Field(default=0.5, gt=0, le=1)
    audio_min_speech_s: float = Field(default=0.3, gt=0, le=1)
    audio_warning_speech_s: float = Field(default=3, gt=0, le=30)
    audio_max_age_s: float = Field(default=15, gt=0, le=60)


class QualityState:
    """Measure one incoming video stream with local monotonic receipt times."""

    def __init__(self, config: QualityConfig, *, clock: Callable[[], float] = time.monotonic):
        self.config, self.clock = config, clock
        self.frames: deque[float] = deque(maxlen=4096)
        self.track_id: str | None = None
        self.started = clock()
        self.last_frame = None
        self.last_sequence = -1
        self.resolution = None
        self.metrics: dict = {}
        self.reset_baseline()

    def reset_baseline(self) -> None:
        self.baseline_fps = None
        self.baseline_pixels = None
        if self.last_frame is not None and self.clock() - self.last_frame < self.config.stall_s:
            width, height = self.resolution
            self.baseline_pixels = width * height
        self.baseline_rates: deque[float] = deque(maxlen=3)

    def frame(self, frame: FrameSignal) -> None:
        now = self.clock()
        if self.track_id is not None and frame.track_id != self.track_id:
            # Keep one stream's cadence when adjacent track callbacks overlap.
            if now - self.last_frame < self.config.stall_s:
                return
            self.frames.clear()
            self.track_id = None
            self.last_sequence = -1
            self.reset_baseline()
        if frame.seq <= self.last_sequence:
            return
        if self.track_id is None:
            self.started = now
        self.track_id = frame.track_id
        self.frames.append(now)
        self.last_frame, self.last_sequence = now, frame.seq
        self.resolution = (frame.width, frame.height)

    def setup_video(self, now: float) -> tuple[float | None, tuple[int, int] | None]:
        if self.last_frame is None or now - self.last_frame >= self.config.stall_s:
            return None, None
        frames = [t for t in self.frames if t >= now - self.config.window_s]
        span = frames[-1] - frames[0] if frames else 0
        fps = (len(frames) - 1) / span if span > 0 else None
        return fps, self.resolution

    def evaluate(self) -> set[str]:
        now, c = self.clock(), self.config
        while self.frames and self.frames[0] < now - c.window_s:
            self.frames.popleft()
        conditions = set()
        fps = None
        if self.last_frame is not None:
            if now - self.last_frame >= c.stall_s:
                conditions.add("low_frame_rate")
                fps = 0.0
            elif now - self.started >= c.window_s:
                fps = len(self.frames) / c.window_s
                if self.baseline_fps is None and fps > 0:
                    self.baseline_rates.append(fps)
                    if len(self.baseline_rates) == 3 and min(self.baseline_rates) >= 0.8 * max(self.baseline_rates):
                        self.baseline_fps = sum(self.baseline_rates) / 3
                if self.baseline_fps is not None and fps <= self.baseline_fps * c.frame_rate_drop_ratio:
                    conditions.add("low_frame_rate")
            if now - self.last_frame < c.stall_s:
                width, height = self.resolution
                pixels = width * height
                if self.baseline_pixels is None:
                    self.baseline_pixels = pixels
                if pixels <= self.baseline_pixels * c.resolution_drop_ratio:
                    conditions.add("low_resolution")
        self.metrics = {"received_fps": fps, "resolution": self.setup_video(now)[1], "baseline_fps": self.baseline_fps}
        return conditions


class QualityFeedback:
    """Own one connection's observations and each capture's warning lifetime."""

    def __init__(
        self,
        pid,
        backend,
        config: QualityConfig,
        *,
        clock=time.monotonic,
        sleep=asyncio.sleep,
        audio_silence_s=0.8,
    ):
        self.pid, self.backend = pid, backend
        self.clock, self.sleep = clock, sleep
        self.config = config
        self.state = QualityState(config, clock=clock)
        self.audio = AudioQuality(config, clock=clock, silence_s=audio_silence_s)
        self.audio_track = None
        self.audio_track_started_us = 0
        self.task: asyncio.Task | None = None

    def frame(self, frame: FrameSignal) -> None:
        self.state.frame(frame)

    def start(self, publisher: AgentRuntime):
        if self.task is not None and not self.task.done():
            raise RuntimeError("previous capture quality task must be drained before restart")
        self.state.reset_baseline()
        self.audio.current = None
        self.audio.bad_speech_s = 0.0
        self.task = asyncio.create_task(self._run(publisher), name=f"capture-quality:{self.pid}")

    def invalidate(self):
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()

    async def stop(self):
        self.invalidate()
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def _record(self, kind, **data):
        await self.backend.capture_quality(self.pid, {"kind": kind, **data})

    async def _run(self, publisher: AgentRuntime):
        source = f"capture-quality:{uuid4().hex}"
        batches: dict[frozenset[str], ResponseOutputs] = {}
        schedule = NotificationSchedule(self.config.repeat_interval_s)
        previous = set()
        try:
            await self._record("monitor_started", policy=self.config.model_dump())
            while True:
                now = self.clock()
                conditions = self.state.evaluate()
                audio_low, audio_metrics = self.audio.evaluate()
                self.state.metrics.update(audio_metrics)
                if audio_low:
                    conditions.add("low_audio_snr")
                if conditions != previous:
                    for batch in list(batches):
                        if not batch <= conditions:
                            await batches[batch].withdraw(self.pid)
                            del batches[batch]
                    await self._record("conditions_changed", conditions=sorted(conditions), metrics=self.state.metrics)
                    previous = set(conditions)
                due = schedule.due(conditions, now)
                if due:
                    # Conditions submitted together share a repeat clock. Replace
                    # their unheard batch, preserving unrelated pending warnings.
                    # Withdrawal never interrupts a warning already playing.
                    for batch in list(batches):
                        if batch.intersection(due):
                            await batches[batch].withdraw(self.pid)
                            del batches[batch]
                    outputs = ResponseOutputs(
                        publisher, VOICE_OUTPUT_TOPIC, source=f"{source}:{','.join(due)}",
                    )
                    batches[frozenset(due)] = outputs
                    ratios = {
                        "low_frame_rate": self.config.frame_rate_drop_ratio,
                        "low_resolution": self.config.resolution_drop_ratio,
                        "low_audio_snr": self.config.audio_snr_drop_ratio,
                    }
                    text = " ".join(
                        _MESSAGES["conditions"][name].format(
                            drop_percent=100 * (1 - ratios[name]),
                            snr_drop_db=audio_metrics["audio_snr_drop_db"],
                        ) for name in due
                    )
                    await outputs.update(self.pid, text)
                    schedule.sent(due, self.clock())
                    await self._record("warning_submitted", conditions=due, text=text)
                await self.sleep(self.config.poll_interval_s)
        except Exception as exc:  # noqa: BLE001 — feedback failure must never stop capture.
            logger.exception("capture quality feedback failed pid={!r}: {}", self.pid, exc)
        finally:
            try:
                for outputs in batches.values():
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
