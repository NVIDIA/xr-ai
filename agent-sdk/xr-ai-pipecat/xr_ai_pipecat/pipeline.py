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

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker

from xr_ai_models import STTService, TTSService
from xr_ai_voicegate import VoiceGateConfig

from .processors.brain import BrainProcessor
from .processors.streaming_tts import StreamingTtsProcessor
from .processors.vad_stt import VadConfig, VadSttProcessor
from .processors.voice_gate import VoiceGateProcessor
from .transport import XRMediaHubTransport


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
