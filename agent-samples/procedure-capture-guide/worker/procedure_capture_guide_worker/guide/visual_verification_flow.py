# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulate fresh visual evidence until a step's confirmation rule is met.

Camera selection and the chosen verifier are injected operations. The flow
preserves frame identity, rejects repeated frames, and owns confirmation state;
Cosmos/CLEF policy and guide state mutation remain outside this component.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from .confirmation import ConfirmationPolicy, ConfirmationWindow
from .flow import FlowUpdate


class FrameIdentity(BaseModel):
    participant_id: str
    track_id: str
    sequence: int
    timestamp_us: int


@dataclass(frozen=True, slots=True)
class Observation:
    available: bool
    value: Any
    frame: FrameIdentity | None = None


@dataclass(frozen=True, slots=True)
class VisualProgress:
    observation: Observation
    hits: int
    fresh: bool


class VisualVerificationFlow:
    def __init__(
        self, *, acquire: Callable[[], Awaitable[Observation]],
        verify: Callable[[Observation], Awaitable[Literal["accepted", "rejected", "unknown"] | None]],
        interval_s: float, required: int, confirmation: ConfirmationPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0 or required < 1:
            raise ValueError("visual interval and required confirmations must be positive")
        if confirmation is not None and confirmation.required != required:
            raise ValueError("confirmation.required must match required")
        self._acquire, self._verify = acquire, verify
        self._interval_s, self._required = interval_s, required
        self._policy, self._clock = confirmation, clock
        self.hits = 0
        self.observed_frames: dict[tuple[str, str], tuple[int, int]] = {}
        self._confirmation = ConfirmationWindow()

    def _reset(self):
        self.hits = 0
        self._confirmation.observations.clear()

    async def poll(self) -> FlowUpdate[VisualProgress]:
        try:
            observation = await self._acquire()
            if not observation.available or observation.frame is None:
                self._reset()
                return FlowUpdate("unavailable", VisualProgress(observation, 0, False), self._interval_s)
            frame = observation.frame
            key, position = (frame.participant_id, frame.track_id), (frame.timestamp_us, frame.sequence)
            previous = self.observed_frames.get(key)
            if previous is not None and position <= previous:
                return FlowUpdate("pending", VisualProgress(observation, self.hits, False), self._interval_s)
            self.observed_frames[key] = position
            verdict = await self._verify(observation)
            if verdict not in {None, "accepted", "rejected", "unknown"}:
                raise ValueError(f"unsupported visual verdict: {verdict!r}")
            if self._policy is not None:
                self.hits = self._confirmation.observe(verdict, self._clock(), self._policy)
            else:
                self.hits = self.hits + 1 if verdict == "accepted" else 0
            status = "unavailable" if verdict is None else "ready" if self.hits >= self._required else "pending"
            return FlowUpdate(status, VisualProgress(observation, self.hits, True), self._interval_s)
        except asyncio.CancelledError:
            self._reset()
            raise
        except Exception:  # noqa: BLE001 — break confirmation before propagating an operation failure.
            self._reset()
            raise
