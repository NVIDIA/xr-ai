# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle host for participant-aware voice handlers."""
from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_models import STTService, TTSService
from xr_ai_voicegate import VoiceGateConfig

from ._handler import VoiceHandler, VoiceTurn
from ._pipeline import _build_voice_pipeline
from ._processors.handler import _VoiceHandlerProcessor
from ._processors.vad_stt import VadConfig
from ._readiness import ProbeFn, wait_for_services
from ._transport import HubVoiceTransport


class VoiceSession:
    """Own readiness, transport, pipeline execution, signals, and cleanup."""

    def __init__(
        self,
        *,
        stt: STTService,
        tts: TTSService,
        vad: VadConfig,
        voice_gate: VoiceGateConfig,
        probes: Mapping[str, ProbeFn] | None = None,
        ready_file: Path | None = None,
        closeables: Iterable[Any] = (),
        text_topic: str = "agent.response",
        idle_timeout_secs: float | None = None,
        transport: HubVoiceTransport | None = None,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.voice_gate = voice_gate
        self.probes = dict(probes or {})
        self.ready_file = ready_file
        self.closeables = tuple(closeables)
        self.text_topic = text_topic
        self.idle_timeout_secs = idle_timeout_secs
        self.transport = transport or HubVoiceTransport()
        self._handler_processor: _VoiceHandlerProcessor | None = None

    async def __aenter__(self) -> "VoiceSession":
        probes = {
            "stt": self.stt.health,
            "tts": self.tts.health,
            **self.probes,
        }
        await wait_for_services(probes)
        if self.ready_file:
            self.ready_file.touch()
        return self

    async def run(
        self,
        handler: VoiceHandler,
        *,
        observer: Callable[[VoiceTurn], Awaitable[None]] | None = None,
        on_participant_joined: Callable[[str], Awaitable[None] | None] | None = None,
        on_participant_left: Callable[[str], Awaitable[None] | None] | None = None,
        on_user_started_speaking: Callable[[str], Awaitable[None] | None] | None = None,
        on_query_superseded: Callable[[str], Awaitable[None] | None] | None = None,
        interrupt_on_supersede: bool = False,
        queue_queries: bool = False,
    ) -> None:
        """Run a voice handler with explicit turn and participant callbacks.

        ``queue_queries`` runs participant queries sequentially instead of
        cancelling the active query. With ``interrupt_on_supersede``, the next
        queued query flushes speech left from the preceding response as it
        starts.
        """
        if self._handler_processor is not None:
            raise RuntimeError("voice session is already running")
        handler_processor = _VoiceHandlerProcessor(
            handler,
            transport=self.transport,
            observer=observer,
            on_participant_joined=on_participant_joined,
            on_participant_left=on_participant_left,
            on_user_started_speaking=on_user_started_speaking,
            on_query_superseded=on_query_superseded,
            interrupt_on_supersede=interrupt_on_supersede,
            queue_queries=queue_queries,
        )
        self._handler_processor = handler_processor
        _, task = _build_voice_pipeline(
            transport=self.transport,
            stt=self.stt,
            tts=self.tts,
            handler_processor=handler_processor,
            vad_cfg=self.vad,
            voice_gate_cfg=self.voice_gate,
            text_topic=self.text_topic,
            idle_timeout_secs=self.idle_timeout_secs,
        )
        loop = asyncio.get_running_loop()
        cancel_requested = False

        def request_cancel() -> None:
            nonlocal cancel_requested
            if cancel_requested:
                return
            cancel_requested = True
            asyncio.create_task(task.cancel())

        installed: list[signal.Signals] = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_cancel)
            except NotImplementedError:
                continue
            installed.append(sig)
        try:
            await PipelineRunner().run(task)
        finally:
            self._handler_processor = None
            for sig in installed:
                loop.remove_signal_handler(sig)

    async def enqueue_query(
        self,
        participant_id: str,
        text: str,
        *,
        fresh_match: bool = False,
        pts_us: int | None = None,
    ) -> None:
        """Submit typed text through the active participant-aware voice path."""
        if self._handler_processor is None:
            raise RuntimeError("voice session is not running")
        await self._handler_processor.enqueue_query(
            participant_id,
            text,
            fresh_match=fresh_match,
            pts_us=pts_us,
        )

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release transport and model clients; safe to call without a context manager."""
        self.transport.shutdown()
        seen: set[int] = set()
        for service in (self.stt, self.tts, *self.closeables):
            if id(service) in seen:
                continue
            seen.add(id(service))
            close = getattr(service, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:
                logger.opt(exception=True).warning("service close failed")
