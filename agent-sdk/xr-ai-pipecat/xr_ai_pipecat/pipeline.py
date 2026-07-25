# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the unified voice pipeline.

One call composes:

    input → VadStt → VoiceGate → brain → StreamingTts → output

and returns the assembled :class:`Pipeline` plus a :class:`PipelineWorker`
ready for :meth:`WorkerRunner.run`. Sample workers do not compose the
pipeline themselves — they subclass :class:`BrainProcessor` and hand it
to this factory.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker
from loguru import logger

from xr_ai_models import STTService, TTSService
from xr_ai_voicegate import VoiceGateConfig

from .processors.brain import BrainProcessor
from .processors.streaming_tts import StreamingTtsProcessor
from .processors.vad_stt import VadConfig, VadSttProcessor
from .processors.voice_gate import VoiceGateProcessor
from .transport import XRMediaHubTransport

_STATUS_REANNOUNCE_INTERVAL_S = 2.0


async def _reannounce_status(transport: XRMediaHubTransport) -> None:
    """Periodically re-send the endpoint's current agent state to clients."""
    while True:
        await asyncio.sleep(_STATUS_REANNOUNCE_INTERVAL_S)
        try:
            await transport.endpoint.republish_statuses()
        except Exception:
            logger.exception("agent-status reannouncement failed")


def make_voice_pipeline(
    *,
    transport: XRMediaHubTransport,
    stt: STTService,
    tts: TTSService,
    brain: BrainProcessor,
    vad_cfg: VadConfig,
    voice_gate_cfg: VoiceGateConfig,
    text_topic: str = "agent.response",
    idle_timeout_secs: float | None = None,
) -> tuple[Pipeline, PipelineWorker]:
    """Assemble the unified voice pipeline.

    The factory builds the :class:`VoiceGateProcessor` first because its
    embedded :class:`xr_ai_voicegate.VoiceGate` is shared with
    :class:`StreamingTtsProcessor` — the TTS processor calls
    ``gate.observe_tts_wav`` so the listening chime gets built at the
    right sample rate.

    ``text_topic`` controls the per-turn data-channel echo emitted by
    :class:`StreamingTtsProcessor`. Set to ``""`` to opt out — samples
    whose brain pushes its own response data message (e.g.
    xr-render-demo) want this off to avoid duplicate sends.

    ``idle_timeout_secs`` controls pipecat's idle-timeout auto-cancel.
    **Disabled by default** (``None``): the pipeline is *never* cancelled for
    inactivity, so a quiet session stays connected indefinitely — important
    for XR sessions where the user may simply not be speaking. Set a positive
    number of seconds to opt in: the worker then cancels the pipeline (and its
    runner) after that long with no user/bot speech. We pass this explicitly
    rather than inheriting pipecat's default, which is ``cancel_on_idle_timeout
    =True`` at ``IDLE_TIMEOUT_SECS`` — i.e. on by default upstream, which would
    silently drop idle sessions.
    """
    voice_gate_proc = VoiceGateProcessor(cfg=voice_gate_cfg, tts=tts)
    streaming_tts   = StreamingTtsProcessor(
        tts        = tts,
        voice_gate = voice_gate_proc.gate,
        transport  = transport,
        text_topic = text_topic,
    )
    vad_stt         = VadSttProcessor(stt=stt, vad_cfg=vad_cfg)

    pipeline = Pipeline([
        transport.input(),
        vad_stt,
        voice_gate_proc,
        brain,
        streaming_tts,
        transport.output(),
    ])
    if idle_timeout_secs is None:
        # Disabled: never cancel the pipeline for inactivity. Override
        # pipecat's on-by-default idle cancel (and the runner cancel) so a
        # quiet XR session is not silently dropped.
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=None,
            cancel_on_idle_timeout=False,
            cancel_runner_on_idle_timeout=False,
        )
    else:
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=idle_timeout_secs,
            cancel_on_idle_timeout=True,
        )
    return pipeline, worker


async def run_voice_pipeline(
    worker: PipelineWorker,
    transport: XRMediaHubTransport,
    *,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Run a voice pipeline and release ``on_ready`` after hub IPC starts.

    The callback is intended for a launcher-managed worker's ready file. It
    runs only after the input transport has entered the processor endpoint
    receive loop. Roster discovery converges asynchronously and does not
    delay process readiness. Once ready, the endpoint publishes an ``idle``
    state and re-announces the current state periodically so late or
    reconnecting clients converge without depending on one initial event.
    If the callback fails, the worker is cancelled and the error propagates
    to the launcher.
    """
    runner_task = asyncio.create_task(
        PipelineRunner().run(worker),
        name="voice-pipeline",
    )
    started_task = asyncio.create_task(
        transport.wait_until_started(),
        name="voice-pipeline-input-start",
    )
    status_task: asyncio.Task | None = None
    try:
        done, _ = await asyncio.wait(
            (runner_task, started_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if runner_task in done:
            await runner_task
            return
        await started_task
        if on_ready:
            on_ready()
        await transport.endpoint.mark_ready()
        status_task = asyncio.create_task(
            _reannounce_status(transport),
            name="voice-pipeline-status",
        )
        await runner_task
    except BaseException:
        if not runner_task.done():
            await worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                _ = await runner_task
        raise
    finally:
        if status_task is not None:
            status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await status_task
        if not started_task.done():
            started_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await started_task
