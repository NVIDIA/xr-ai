# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BrainProcessor`` — base class for sample-specific reasoning.

Sample brains subclass this and implement :meth:`handle_query`. The
base class owns:

- the per-pid in-flight task,
- cancellation on a new ``GatedQueryFrame`` (a fresh user query
  supersedes any prior in-flight response) and on ``InterruptionFrame``
  (explicit stop, e.g. from the voice gate),
- pushing each yielded token/chunk as a downstream ``TextFrame``,
- the optional participant join/leave / user-started-speaking lifecycle
  hooks.

``UserStartedSpeakingFrame`` is a hook only — it does NOT cancel
in-flight work. Cancelling on every speech onset interrupts the agent
mid-sentence the moment the user starts a follow-up; worse, any AEC
leak of the agent's own TTS makes the agent cancel itself. The voice
gate emits ``InterruptionFrame`` explicitly when the user actually
says "stop"; that is the right cancel signal.

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
        # Joined pids so the user-speech hook can fire on the cold path
        # (no in-flight task yet) — useful for camera warmup. Cleared on
        # ``ParticipantLeftFrame``.
        self._joined: set[str] = set()

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

        Useful for speculative warmup (e.g. camera, image fetch) so the
        next query starts with a hot cache. Does NOT cancel the
        in-flight query — cancellation happens on the next
        ``GatedQueryFrame`` or explicit ``InterruptionFrame``. Default:
        no-op.
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
            # Hook only — speech onset is NOT a cancel signal. Samples
            # may override on_user_started_speaking for speculative work
            # (e.g. camera warmup); cancellation happens on the next
            # GatedQueryFrame or on an explicit InterruptionFrame.
            await self._fan_out_user_started_speaking()
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
            self._joined.add(frame.participant_id)
            await self.on_participant_joined(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            self._joined.discard(frame.participant_id)
            await self.on_participant_left(frame.participant_id)
            self._cancel_pid(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    async def _fan_out_user_started_speaking(self) -> None:
        # The pipecat ``UserStartedSpeakingFrame`` carries no pid, so we
        # fan the hook out across every pid the brain currently knows
        # about. Use the joined set, not the in-flight tasks: the cold
        # path (first utterance, nothing in flight yet) is exactly where
        # speculative warmup matters most. Single-speaker samples
        # collapse to one call.
        for pid in list(self._joined):
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
