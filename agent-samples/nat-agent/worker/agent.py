# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NatAgent — assembles the full nat-agent worker.

* Constructs the ``XRMediaHubTransport`` (audio + data IPC; video filtered out).
* Builds the Pipecat pipeline: input → stt → nat → tts → output.
* Wires the data-channel direct path (typed text + ``ping:`` rewrite) outside
  the Pipecat pipeline so it can fire NAT inferences without an audio frame.
* Tracks the current participant lazily on first inbound traffic.
* Pre-warms the NAT workflow in the background so the first user turn doesn't
  pay the workflow-build cost.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from xr_ai_agent import (
    AudioChunk, DataMessage, ParticipantEvent, ProcessorEndpoint,
)

from audio import stream_sentences_to_audio
from config import WorkerConfig
from nat_backend import NatBackend
from processors import build_pipeline
from services import SttClient, TtsClient
from transport import XRMediaHubTransport

log = logging.getLogger("nat_agent")


def _now_us() -> int:
    return time.time_ns() // 1_000


# Topics we publish ourselves — never feed them back into NAT.
_AGENT_OUTBOUND_TOPICS = frozenset({
    "agent.response", "stt.partial", "stt.transcript",
})


class NatAgent:
    """The whole worker, minus the entry-point boilerplate.

    Owns the transport, the Pipecat pipeline runner, the NAT backend, and
    the side-channel data handler.
    """

    def __init__(
        self,
        cfg: WorkerConfig,
        stt: SttClient,
        tts: TtsClient,
        nat: NatBackend,
    ) -> None:
        self._cfg = cfg
        self._stt = stt
        self._tts = tts
        self._nat = nat

        self._transport = XRMediaHubTransport()
        self._pipeline, self._task = build_pipeline(
            self._transport, self._stt, self._tts, self._nat, cfg,
        )

        self._transport.endpoint.on_data(self._on_data)
        self._transport.endpoint.on_audio(self._on_audio_chunk)
        self._transport.endpoint.on_participant(self._on_participant)

        self._prewarm_task: asyncio.Task | None = None
        self._runner_task: asyncio.Task | None = None

        # Latest-only-replace queue for data-channel queries — same shape as
        # NatProcessor's voice queue but in this object's loop. Without
        # this, rapid pings (each fires a 5-15 s NAT turn) all serialise
        # through NatBackend's _infer_lock and pile up unbounded; the user
        # can never get fresh results because the queue head is always
        # stale. Bounded at one in-flight + one pending; a newer ping
        # replaces the pending one and the in-flight result is dropped if
        # superseded.
        self._data_pending: tuple[str, str, int] | None = None
        self._data_lock = asyncio.Lock()
        self._data_drain_task: asyncio.Task | None = None

    # ── target participant lazy bind ──────────────────────────────────────────

    def _set_target_if_absent(self, pid: str) -> None:
        if not pid:
            return
        if self._transport.target_participant == pid:
            return
        log.info("target participant set from traffic: %r", pid)
        self._transport.set_target_participant(pid)

    async def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        self._set_target_if_absent(chunk.participant_id)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            log.info("participant joined: %r", event.participant_id)
            self._set_target_if_absent(event.participant_id)
            await self._transport.endpoint.set_status(
                "idle", event.participant_id,
            )
        else:
            log.info("participant left: %r", event.participant_id)
            self._transport.cleanup_participant(event.participant_id)

    # ── data-channel direct path (typed text / ping) ──────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        # Don't bounce our own outbound topics back through NAT.
        if msg.topic in _AGENT_OUTBOUND_TOPICS:
            return

        self._set_target_if_absent(msg.participant_id)

        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                query = payload.get("query", "")
            else:
                query = str(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            query = msg.data.decode(errors="replace")

        if not query:
            return

        pid = msg.participant_id

        # Web client's "describe what you see" button sends "ping:..." —
        # rewrite to a default visual prompt so the LLM picks ask_image.
        if query.startswith("ping:") or query.lower().strip() == "ping":
            query = "Describe what you see."
            log.info("ping  pid=%r  rewritten to describe-scene", pid)
        else:
            log.info("data  pid=%r  query=%r", pid, query[:80])

        # ``msg.pts_us`` is the moment the data message was sent — the
        # closest analogue we have to "when the user asked" on this path.
        # Fall back to a fresh wall-clock reading when the producer didn't
        # populate it (legacy or non-pipecat clients).
        ref_us = msg.pts_us if msg.pts_us > 0 else _now_us()

        async with self._data_lock:
            superseded = self._data_pending is not None
            self._data_pending = (query, pid, ref_us)
            if superseded:
                log.info("data pending REPLACED — older query dropped, "
                         "new=%r", query[:80])
            else:
                log.info("data queued  %r", query[:80])

            if self._data_drain_task is None or self._data_drain_task.done():
                self._data_drain_task = asyncio.create_task(
                    self._data_drain(), name="nat-data-drain",
                )

    async def _data_drain(self) -> None:
        """Process queued data-channel queries one at a time, latest-wins.

        Mirrors NatProcessor._drain (same pattern; see processors.py for the
        full rationale). The currently running NAT inference is not
        cancelled — its result is silently discarded if a newer query
        arrives in the meantime.
        """
        loop = asyncio.get_running_loop()
        announced_pid: str | None = None
        try:
            while True:
                async with self._data_lock:
                    if self._data_pending is None:
                        return
                    query, pid, ref_us = self._data_pending
                    self._data_pending = None

                if pid and pid != announced_pid:
                    await self._transport.endpoint.set_status("processing", pid)
                    announced_pid = pid

                log.info(
                    "data inference START  pid=%r  ref=%d  query=%r",
                    pid, ref_us, query[:80],
                )
                try:
                    answer = await loop.run_in_executor(
                        None, self._nat.infer, query, pid, ref_us,
                    )
                except Exception:
                    log.exception("data inference failed  query=%r", query[:80])
                    await self._respond(pid, "Sorry, something went wrong.")
                    continue

                async with self._data_lock:
                    if self._data_pending is not None:
                        log.info(
                            "data result DISCARDED — newer query "
                            "pending  stale_query=%r",
                            query[:80],
                        )
                        continue

                log.info("data response  pid=%r  %d chars  text=%r",
                         pid, len(answer), answer[:200])
                await self._respond(pid, answer)
        finally:
            if announced_pid:
                try:
                    await self._transport.endpoint.set_status("idle", announced_pid)
                except Exception:
                    pass  # transport already closed during pipeline shutdown

    async def _respond(self, pid: str, text: str) -> None:
        """Publish *text* as both a data message and TTS audio."""
        if not text or not pid:
            return

        await self._transport.send_return_data(DataMessage(
            participant_id=pid,
            topic="agent.response",
            pts_us=_now_us(),
            data=text.encode(),
        ))

        try:
            await stream_sentences_to_audio(
                self._transport.endpoint, self._tts.synthesize, text, pid,
            )
        except Exception:
            log.exception("tts stream failed  pid=%r", pid)

    # ── pre-warm ──────────────────────────────────────────────────────────────

    async def _prewarm(self) -> None:
        """Build the NAT workflow up-front. LLM server is already healthy."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._nat.ensure_loaded)
            log.info("NAT workflow ready (pre-warmed)")
        except Exception:
            log.exception("NAT workflow build failed; will retry on first turn")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        from pipecat.pipeline.runner import PipelineRunner

        self._prewarm_task = asyncio.create_task(
            self._prewarm(), name="nat-prewarm",
        )
        runner = PipelineRunner()
        self._runner_task = asyncio.create_task(
            runner.run(self._task), name="pipecat-runner",
        )
        await self._runner_task

    def shutdown(self) -> None:
        """Cancel background tasks and tear down the transport.

        Synchronous so it can be invoked from a signal handler.
        """
        if self._prewarm_task and not self._prewarm_task.done():
            self._prewarm_task.cancel()
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
        if self._data_drain_task and not self._data_drain_task.done():
            self._data_drain_task.cancel()
        # Schedule async tear-down without blocking the signal handler.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._stt.close())
            loop.create_task(self._tts.close())
            loop.create_task(self._nat.close())
        except RuntimeError:
            pass
        self._transport.shutdown()
