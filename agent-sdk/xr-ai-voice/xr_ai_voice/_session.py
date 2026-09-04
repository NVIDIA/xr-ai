# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media lifecycle for the runtime voice agent."""
from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_hub import ProcessorEndpoint
from xr_ai_models import STTService, TTSService
from xr_ai_voicegate import VoiceGateConfig

from ._pipeline import _build_voice_pipeline
from ._processors.io import _VoiceIOProcessor
from ._processors.vad_stt import VadConfig
from ._readiness import ProbeFn, wait_for_services
from ._transport import HubVoiceTransport
from ._types import VoiceInputSink, VoiceResponse

_STATUS_REANNOUNCE_INTERVAL_S = 2.0


async def _reannounce_status(transport: HubVoiceTransport) -> None:
    """Periodically re-send the endpoint's current agent state to clients."""
    while True:
        await asyncio.sleep(_STATUS_REANNOUNCE_INTERVAL_S)
        try:
            await transport.endpoint.republish_statuses()
        except Exception:
            logger.opt(exception=True).warning("agent-status reannouncement failed")


class _VoiceSession:
    """Private media engine owned by :class:`VoiceAgent`."""

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
        self._transport = transport
        self._io_processor: _VoiceIOProcessor | None = None
        self._closed = False

    @property
    def transport(self) -> HubVoiceTransport:
        """Return the transport, constructing the default only when needed."""
        if self._transport is None:
            self._transport = HubVoiceTransport()
        return self._transport

    @property
    def endpoint(self) -> ProcessorEndpoint:
        """Return the hub endpoint after readiness has initialized the transport."""

        if self._transport is None:
            raise RuntimeError("voice session endpoint is not ready")
        return self._transport.endpoint

    @property
    def is_running(self) -> bool:
        """Whether the session currently accepts voice or text queries."""
        return self._io_processor is not None

    async def __aenter__(self) -> "_VoiceSession":
        if self._closed:
            raise RuntimeError("voice session is closed")
        probes = {
            "stt": self.stt.health,
            "tts": self.tts.health,
            **self.probes,
        }
        try:
            await wait_for_services(probes)
            _ = self.transport
        except BaseException:
            await self.close()
            raise
        return self

    async def run(
        self,
        input_sink: VoiceInputSink,
        *,
        on_transcript: Callable[[str, str, int], Awaitable[None]] | None = None,
        on_participant_joined: Callable[[str], Awaitable[None] | None] | None = None,
        on_participant_left: Callable[[str], Awaitable[None] | None] | None = None,
        on_speech_started: Callable[[str], Awaitable[None] | None] | None = None,
        on_speech_stopped: Callable[[str], Awaitable[None] | None] | None = None,
        on_interrupted: Callable[[str | None], Awaitable[None] | None] | None = None,
        interrupt_on_supersede: bool = False,
    ) -> None:
        """Run media input/output until the pipeline exits."""
        if self._io_processor is not None:
            raise RuntimeError("voice session is already running")
        io_processor = _VoiceIOProcessor(
            input_sink,
            transport=self.transport,
            on_participant_joined=on_participant_joined,
            on_participant_left=on_participant_left,
            on_speech_started=on_speech_started,
            on_speech_stopped=on_speech_stopped,
            on_interrupted=on_interrupted,
            interrupt_on_supersede=interrupt_on_supersede,
        )
        self._io_processor = io_processor
        _, task = _build_voice_pipeline(
            transport=self.transport,
            stt=self.stt,
            tts=self.tts,
            io_processor=io_processor,
            vad_cfg=self.vad,
            voice_gate_cfg=self.voice_gate,
            on_final_transcript=on_transcript,
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
        runner_task = asyncio.create_task(
            PipelineRunner().run(task),
            name="voice-session-pipeline",
        )
        started_task = asyncio.create_task(
            self.transport.wait_until_started(),
            name="voice-session-input-start",
        )
        status_task: asyncio.Task | None = None
        try:
            done, _ = await asyncio.wait(
                (runner_task, started_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if runner_task in done:
                _ = await runner_task
                return
            _ = await started_task
            if self.ready_file:
                self.ready_file.touch()
            await self.transport.endpoint.mark_ready()
            status_task = asyncio.create_task(
                _reannounce_status(self.transport),
                name="voice-session-status",
            )
            _ = await runner_task
        except BaseException:
            if not runner_task.done():
                await task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    _ = await runner_task
            raise
        finally:
            if status_task is not None:
                status_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    _ = await status_task
            if not started_task.done():
                started_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                _ = await started_task
            self._io_processor = None
            for sig in installed:
                loop.remove_signal_handler(sig)

    async def enqueue_query(
        self,
        participant_id: str,
        text: str,
        *,
        pts_us: int | None = None,
    ) -> None:
        """Submit typed text through the active participant-aware voice path."""
        if self._io_processor is None:
            raise RuntimeError("voice session is not running")
        await self._io_processor.enqueue_query(
            participant_id,
            text,
            pts_us=pts_us,
        )

    async def enqueue_response(
        self,
        participant_id: str,
        response: VoiceResponse,
        *,
        interrupt: bool = False,
        pts_us: int | None = None,
    ) -> None:
        """Queue finite or incremental output on the active participant voice path."""

        if self._io_processor is None:
            raise RuntimeError("voice session is not running")
        await self._io_processor.enqueue_response(
            participant_id,
            response,
            interrupt=interrupt,
            pts_us=pts_us,
        )

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release transport and model clients; safe to call without a context manager."""
        if self._closed:
            return
        self._closed = True
        if self._transport is not None:
            self._transport.shutdown()
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
