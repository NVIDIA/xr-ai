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
from typing import TYPE_CHECKING, AsyncIterator, Union

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..frames import (
    BrainResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
)

if TYPE_CHECKING:
    from ..transport import XRMediaHubTransport


QueryResult = Union[AsyncIterator[str], str]


class BrainProcessor(FrameProcessor):
    """Subclass and implement :meth:`handle_query`.

    The base class also forwards every non-handled frame, so this
    processor can sit anywhere in the chain without dropping
    pipecat-internal traffic.
    """

    def __init__(self, *, transport: "XRMediaHubTransport | None" = None) -> None:
        super().__init__()
        self._inflight: dict[str, asyncio.Task] = {}
        # Pids that have had at least one query — gates the supersede
        # hook on every non-first query for the pid, regardless of
        # whether the prior brain task is still running. The brain-task
        # status is the wrong gate: by the time a follow-up query lands,
        # the prior brain task may already be done while the downstream
        # TTS audio is still streaming out. Cleared on
        # ``ParticipantLeftFrame``.
        self._seen_query: set[str] = set()
        # Joined pids so the user-speech hook can fire on the cold path
        # (no in-flight task yet) — useful for camera warmup. Cleared on
        # ``ParticipantLeftFrame``.
        self._joined: set[str] = set()
        # Optional — when supplied, the output transport is steered at
        # the first ``ParticipantJoinedFrame`` so the single-participant
        # return-audio / return-data path "just works" without per-sample
        # wiring. Samples that handle multi-participant routing manually
        # leave this unset.
        self._transport = transport

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
        """Override to react at the leading edge of a speech burst.

        Fires when VAD accumulates enough speech to cross ``min_speech``
        (default 0.15 s) — before STT has produced a transcript, and before
        the voice gate has decided whether to pass the utterance through.

        ``pid`` is the participant who started speaking.

        **Wake-word mode:** this hook fires on *every* speech onset,
        regardless of gate state.  ``VoiceGateProcessor`` forwards
        ``UserStartedSpeakingFrame`` unconditionally — it does not suppress
        the frame when the gate is closed.  Do not assume the user is
        addressing the agent; treat this as a speculative signal only.

        **What to do here:** pre-fetch data that the next query is likely
        to need (scene state, retrieval cache, model warm-up).

        Does NOT cancel any in-flight brain task — cancellation happens on
        the next ``GatedQueryFrame`` or an explicit ``InterruptionFrame``.
        Exceptions from this hook are logged and swallowed.  Default: no-op.
        """
        return

    async def on_query_superseded(self, pid: str) -> None:
        """Override to react when a new query supersedes the prior one.

        Fires on every ``GatedQueryFrame`` after the first for ``pid``,
        regardless of whether the prior brain task is still in-flight.
        Audio from the prior response may still be playing even if that
        task already completed — sample brains push ``InterruptionFrame``
        to drain the TTS sender + flush the hub buffer.

        The brain-task status is the wrong gate. Cancelling the brain
        task only stops *new* ``TextFrame``s from this processor; the
        streaming TTS sender queue, the hub's pacing pipe, and any
        jitter buffer downstream continue to deliver whatever was
        already enqueued. If the prior task finished quickly (e.g. a
        short VLM response that streamed in under a second) the TTS
        audio for that response can still be mid-flight when the
        follow-up query arrives. Firing on every non-first query lets
        the sample decide what to do about prior state (audio drain,
        accumulated UI, etc.).

        The library still cancels any prior in-flight task immediately
        after this hook returns — that is not configurable (you cannot
        have two queries in flight for the same pid). Default: no-op —
        audio continues, the new response queues behind it.

        Exceptions from this hook are logged and swallowed; the
        supersede + spawn of the new query proceeds either way.
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
            # Hook only — speech onset is NOT a cancel signal.
            # transport_source carries the pid when set by VadSttProcessor.
            await self._dispatch_user_started_speaking(frame.transport_source)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                # Cancel only the participant who triggered the interruption.
                logger.info("brain cancel pid={!r} reason=interruption", pid)
                self._cancel_pid(pid)
            else:
                # No pid on frame — global interrupt (legacy / unknown source).
                if self._inflight:
                    for p in list(self._inflight):
                        logger.info("brain cancel pid={!r} reason=interruption", p)
                self._cancel_all_inflight()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, GatedQueryFrame):
            await self._spawn_query(frame)
            return

        if isinstance(frame, ParticipantJoinedFrame):
            self._joined.add(frame.participant_id)
            logger.info("brain participant joined pid={!r}", frame.participant_id)
            # Single-participant default: steer the output transport at
            # the first join so return-audio / return-data routing works
            # without per-sample wiring. Samples that need multi-pid
            # routing construct the brain without a transport.
            if self._transport is not None:
                self._transport.set_target_participant(frame.participant_id)
            await self.on_participant_joined(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            self._joined.discard(frame.participant_id)
            self._seen_query.discard(frame.participant_id)
            logger.info("brain participant left pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.cleanup_participant(frame.participant_id)
            await self.on_participant_left(frame.participant_id)
            self._cancel_pid(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    async def _dispatch_user_started_speaking(self, pid: str | None) -> None:
        # VadSttProcessor sets frame.transport_source = pid, so we normally
        # get an exact pid. Fall back to all joined pids only when the frame
        # arrives without transport_source (e.g. from a third-party processor).
        targets = [pid] if (pid and pid in self._joined) else list(self._joined)
        for p in targets:
            try:
                await self.on_user_started_speaking(p)
            except Exception:
                logger.exception("on_user_started_speaking raised pid={!r}", p)

    async def _spawn_query(self, frame: GatedQueryFrame) -> None:
        pid = frame.participant_id
        logger.info(
            "brain dispatch pid={!r} fresh_match={}", pid, frame.fresh_match,
        )
        # A fresh query supersedes the previous one for the same pid.
        # Fire the supersede hook on every non-first query for the pid,
        # regardless of whether the prior brain task is still running.
        # The brain-task status is the wrong gate: a short prior
        # response may have finished assembly while its TTS audio is
        # still streaming out, and the sample needs to decide what to
        # do about that (e.g. push InterruptionFrame to drain the TTS
        # sender + flush the hub buffer). Fire *before* cancelling so
        # the override lands while the prior task's downstream state is
        # still coherent.
        if pid in self._seen_query:
            logger.info("brain superseded pid={!r}", pid)
            try:
                await self.on_query_superseded(pid)
            except Exception:
                logger.exception("on_query_superseded raised pid={!r}", pid)
        self._seen_query.add(pid)
        # Library still owns cancelling any task that IS still running.
        self._cancel_pid(pid)
        self._inflight[pid] = asyncio.create_task(
            self._run_query(frame), name=f"brain-query-{pid}",
        )

    async def _run_query(self, frame: GatedQueryFrame) -> None:
        pid = frame.participant_id
        # Accumulate every token/string we yield so the trailing
        # ``BrainResponseEndFrame`` carries the full assembled response.
        # Downstream ``StreamingTtsProcessor`` uses this to send a single
        # data-channel echo per turn, matching pre-migration behavior.
        accumulated: list[str] = []
        cancelled = False
        try:
            result = await self.handle_query(pid, frame.text, frame.fresh_match)
            if isinstance(result, str):
                if result:
                    accumulated.append(result)
                    await self._push_text(result, pid=pid)
                return  # finally still fires — emits BrainResponseEndFrame
            async for chunk in result:
                if not chunk:
                    continue
                accumulated.append(chunk)
                await self._push_text(chunk, pid=pid)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("brain handle_query raised pid={!r}", pid)
        finally:
            # Emit the end marker only when the turn completed
            # (success OR caught exception). On asyncio cancellation we
            # leave the marker out — a cancel means the user superseded
            # the response, and a half-assembled data echo would
            # contradict the new turn that's about to render.
            if not cancelled:
                logger.info("brain query complete pid={!r}", pid)
                try:
                    await self.push_frame(BrainResponseEndFrame(
                        pid    = pid,
                        text   = "".join(accumulated),
                        pts_us = frame.pts_us,
                    ))
                except Exception:
                    logger.exception("emit BrainResponseEndFrame failed pid={!r}", pid)

            # Don't pop if a newer task has taken our slot.
            current = self._inflight.get(pid)
            if current is asyncio.current_task():
                self._inflight.pop(pid, None)

    async def _push_text(self, text: str, *, pid: str) -> None:
        """Push a ``TextFrame`` tagged with the participant id.

        ``transport_destination`` flows through the pipeline to the
        ``StreamingTtsProcessor``, which copies it onto the
        ``OutputAudioRawFrame``s it emits so the output transport knows
        which participant to address. Without this tag, the empty
        string ends up on every downstream send and the hub drops the
        audio.
        """
        f = TextFrame(text=text)
        f.transport_destination = pid
        await self.push_frame(f)

    def _cancel_pid(self, pid: str) -> None:
        task = self._inflight.pop(pid, None)
        if task is not None and not task.done():
            task.cancel()

    def _cancel_all_inflight(self) -> None:
        for pid, task in list(self._inflight.items()):
            if not task.done():
                task.cancel()
        self._inflight.clear()
