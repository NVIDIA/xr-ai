# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BrainProcessor`` — base class for sample-specific reasoning.

Sample brains subclass this and implement :meth:`handle_query`. The
base class owns:

- the per-pid in-flight task,
- cancellation on ``UserStartedSpeakingFrame`` and ``InterruptionFrame``
  (the user is talking again — abandon the old response),
- pushing each yielded token/chunk as a downstream ``TextFrame``,
- the optional participant join/leave lifecycle hooks.

Sample brains are tiny: write the reasoning loop and (optionally) any
per-pid setup/teardown.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Union

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..frames import GatedQueryFrame, ParticipantJoinedFrame, ParticipantLeftFrame


QueryResult = Union[AsyncIterator[str], str]


class BrainProcessor(FrameProcessor):
    """Subclass and implement :meth:`handle_query`.

    The base class also forwards every non-handled frame, so this
    processor can sit anywhere in the chain without dropping
    pipecat-internal traffic.
    """

    def __init__(self) -> None:
        super().__init__()
        self._inflight: dict[str, asyncio.Task] = {}

    # ── overrides ─────────────────────────────────────────────────────────────

    async def handle_query(
        self, pid: str, text: str, fresh_match: bool,
    ) -> QueryResult:
        """Run the brain's reasoning for a single query.

        Return either a single string (one TextFrame downstream) or an
        async iterator of strings (one TextFrame per yielded chunk).
        """
        raise NotImplementedError

    async def on_user_started_speaking(self, pid: str) -> None:
        """Override for sample-specific behavior on speech start.

        Fires before the in-flight query (if any) is cancelled — useful
        for speculative warmup (e.g. camera, image fetch) so the next
        query starts with a hot cache. Default: no-op.
        """
        return

    async def on_participant_joined(self, pid: str) -> None:
        """Override for per-pid setup. Default: no-op."""
        return

    async def on_participant_left(self, pid: str) -> None:
        """Override for per-pid teardown. Default: no-op."""
        return

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # super().process_frame fires base-class interruption + metrics
        # plumbing for InterruptionFrame; we still cancel our task
        # ourselves because the base class does not know about it.
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self._on_user_started_speaking_locally()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._cancel_all_inflight()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, GatedQueryFrame):
            await self._spawn_query(frame)
            return

        if isinstance(frame, ParticipantJoinedFrame):
            await self.on_participant_joined(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            await self.on_participant_left(frame.participant_id)
            self._cancel_pid(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    async def _on_user_started_speaking_locally(self) -> None:
        # The frame itself doesn't carry pid; cancel everything and fan
        # the hook out across the pids we currently know about. Most
        # samples are single-speaker today so this collapses to a
        # single cancel.
        pids = list(self._inflight)
        self._cancel_all_inflight()
        for pid in pids:
            try:
                await self.on_user_started_speaking(pid)
            except Exception:
                logger.exception("on_user_started_speaking raised pid={!r}", pid)

    async def _spawn_query(self, frame: GatedQueryFrame) -> None:
        pid = frame.participant_id
        # A fresh query supersedes any previous in-flight reasoning for
        # the same pid — happens when the user squeezes in a follow-up
        # before the last response completes.
        self._cancel_pid(pid)
        self._inflight[pid] = asyncio.create_task(
            self._run_query(frame), name=f"brain-query-{pid}",
        )

    async def _run_query(self, frame: GatedQueryFrame) -> None:
        try:
            result = await self.handle_query(
                frame.participant_id, frame.text, frame.fresh_match,
            )
            if isinstance(result, str):
                if result:
                    await self.push_frame(TextFrame(text=result))
                return
            async for chunk in result:
                if not chunk:
                    continue
                await self.push_frame(TextFrame(text=chunk))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("brain handle_query raised pid={!r}", frame.participant_id)
        finally:
            # Don't pop if a newer task has taken our slot.
            current = self._inflight.get(frame.participant_id)
            if current is asyncio.current_task():
                self._inflight.pop(frame.participant_id, None)

    def _cancel_pid(self, pid: str) -> None:
        task = self._inflight.pop(pid, None)
        if task is not None and not task.done():
            task.cancel()

    def _cancel_all_inflight(self) -> None:
        for pid, task in list(self._inflight.items()):
            if not task.done():
                task.cancel()
        self._inflight.clear()
