# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own bounded/coalesced semantic policy scheduling inside xr_ai_voice."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .._frames import (
    GatedQueryFrame,
    ParticipantLeftFrame,
    _TurnInterruptFrame,
    _TurnResetFrame,
    _TurnTranscriptFrame,
)
from .._turn_policy import VoiceOutputState, VoiceTurnAction, VoiceTurnPolicy

_MAX_SOURCES = 64


class _TurnPolicyProcessor(FrameProcessor):
    def __init__(
        self, policy: VoiceTurnPolicy, *, authorize: Callable[[str, object | None], bool] | None = None
    ) -> None:
        super().__init__()
        if not math.isfinite(policy.settle_s) or policy.settle_s < 0:
            raise ValueError("turn policy settle_s must be finite and nonnegative")
        self._policy = policy
        self._authorize = authorize
        self._authorities: dict[str, object | None] = {}
        self._epochs: dict[str, str] = {}
        self._identities: dict[str, object] = {}
        self._requests: dict[str, str] = {}
        self._dirty: OrderedDict[str, float] = OrderedDict()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._closed = False

    def _mark(self, pid: str, *, settle: bool = False) -> None:
        self._dirty[pid] = asyncio.get_running_loop().time() + (self._policy.settle_s if settle else 0)
        self._wake.set()

    def output(self, state: VoiceOutputState) -> None:
        if self._closed or state.participant_id not in self._epochs:
            return
        if self._requests.get(state.participant_id) != state.request_id:
            return
        try:
            self._policy.output(state)
        except Exception:  # noqa: BLE001 - synchronous user callback isolation
            # isolation boundary: revoke this source after a failed notification.
            logger.exception("semantic voice output callback failed pid={!r}", state.participant_id)
            self._reset(state.participant_id)
            return
        self._mark(state.participant_id)
        if state.stage in {"done", "interrupted"}:
            self._requests.pop(state.participant_id, None)

    def _reset(self, pid: str) -> None:
        self._epochs.pop(pid, None)
        self._authorities.pop(pid, None)
        self._identities.pop(pid, None)
        self._dirty.pop(pid, None)
        self._requests.pop(pid, None)
        # Removing an epoch revokes in-flight work. No tombstone grows on churn.
        try:
            self._policy.reset(pid)
        except Exception:  # noqa: BLE001 - synchronous user callback isolation
            # isolation boundary: SDK authority is revoked even if policy cleanup fails.
            logger.exception("semantic voice reset callback failed pid={!r}", pid)
        self._wake.set()

    async def _shutdown(self) -> None:
        self._closed = True
        for pid in tuple(self._epochs):
            self._reset(pid)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def cleanup(self) -> None:
        await self._shutdown()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._worker = asyncio.create_task(self._run(), name="voice-turn-policy")
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._shutdown()
        elif isinstance(frame, (_TurnResetFrame, ParticipantLeftFrame)):
            self._reset(frame.participant_id)
        elif isinstance(frame, _TurnTranscriptFrame):
            update = frame.update
            pid = update.participant_id
            if self._closed:
                return
            if self._authorize is not None and not self._authorize(pid, frame.authority):
                return
            if self._epochs.get(pid) != update.source_epoch or self._authorities.get(pid) is not frame.authority:
                if pid not in self._epochs and len(self._epochs) >= _MAX_SOURCES:
                    logger.error("semantic voice source capacity exceeded")
                    return
                self._reset(pid)
                self._epochs[pid] = update.source_epoch
                self._authorities[pid] = frame.authority
                self._identities[pid] = object()
            try:
                self._policy.update(update)
            except Exception:  # noqa: BLE001 - synchronous user callback isolation
                # isolation boundary: invalid input state cannot admit a query.
                logger.exception("semantic voice update callback failed pid={!r}", pid)
                self._reset(pid)
                return
            self._mark(pid, settle=True)
            return
        elif isinstance(frame, InterruptionFrame):
            # Gate-owned STOP/release revokes pending semantic work too.
            for pid in [frame.transport_source] if frame.transport_source else tuple(self._epochs):
                self._reset(pid)
        await self.push_frame(frame, direction)

    async def _run(self) -> None:
        while not self._closed:
            if not self._dirty:
                self._wake.clear()
                await self._wake.wait()
                continue
            pid = min(self._dirty, key=self._dirty.__getitem__)
            delay = self._dirty[pid] - asyncio.get_running_loop().time()
            if delay > 0:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), delay)
                except TimeoutError:
                    pass
                continue
            self._dirty.pop(pid)
            identity = self._identities.get(pid)
            authority = self._authorities.get(pid)
            try:
                actions = await self._policy.decide(pid)
                if self._identities.get(pid) is not identity:
                    continue
                if self._authorize is not None and not self._authorize(pid, authority):
                    self._reset(pid)
                    continue
                for action in actions:
                    if self._closed or self._identities.get(pid) is not identity:
                        break
                    if self._authorize is not None and not self._authorize(pid, authority):
                        self._reset(pid)
                        break
                    if not isinstance(action, VoiceTurnAction):
                        raise TypeError("turn policy must return VoiceTurnAction values")
                    if action.kind == "interrupt":
                        if self._requests.get(pid) != action.request_id:
                            continue
                        self._requests.pop(pid, None)
                        stop = _TurnInterruptFrame(request_id=action.request_id)
                        stop.transport_source = pid
                        # Deliberately goes downstream: this is admission policy,
                        # not a gate reset that would discard the new follow-up.
                        await self.push_frame(stop)
                    else:
                        self._requests[pid] = action.request_id
                        await self.push_frame(
                            GatedQueryFrame(
                                participant_id=pid,
                                text=action.text,
                                fresh_match=False,
                                pts_us=action.timestamp_us,
                                request_id=action.request_id,
                                _turn_authority=authority,
                            )
                        )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - owned policy/task supervision
                # isolation boundary: optional policy failure cannot admit input.
                logger.exception("semantic voice policy failed pid={!r}", pid)
