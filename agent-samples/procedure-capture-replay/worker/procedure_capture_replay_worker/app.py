# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run procedure capture or guided replay with shared speech and vision services.

Called by __main__.py, this module sets up the model clients and hub connection,
then brings VoiceAgent and the selected capability agent together in one runtime.
This module coordinates their startup, readiness, and cleanup without starting
the other capability's background work.

Refer to capture/agent.py for capture command handling and replay/agent.py for
guide execution and verification.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from xr_ai_hub import ProcessorEndpoint
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool, MultiImageQueryTool
from xr_ai_voice import VadConfig, VoiceAgent, VoiceAggregationAgent
from xr_ai_voicegate import load_voice_gate_config

from .capture.agent import CAPTURE_INPUT_TOPIC, CaptureAgent
from .capture.capturing import _Capture
from .config import WorkerConfig
from .replay.agent import ReplayAgent
from .replay.events import INTERRUPTED_TOPIC, PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, USER_QUERY_TOPIC
from .replay.guides import select_guide

_HUB_PUB = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker")
    # Validate approval and pin guide bytes before opening model or hub clients.
    guide = None
    if config.mode == "replay":
        if config.replay is None:
            raise ValueError("replay mode requires replay configuration")
        guide = select_guide(config.replay.guides_dir, config.replay.guide_name)
    elif config.capture is None:
        raise ValueError("capture mode requires capture configuration")
    models = load_models_config(config.models_config)
    tts, vlm = make_tts(models, "tts"), make_vlm(models, "vlm")
    llm = make_llm(models, "llm") if guide is not None else None
    # Capture must finish draining after voice stops. Its frame and participant
    # endpoint therefore has a lifetime controlled by this composition layer.
    endpoint = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=endpoint, images=images, frame_max_age_s=config.frame_max_age_s, frame_timeout_s=config.frame_timeout_s
    )
    runtime = AgentRuntime()
    capture = replay = aggregation = None
    voice_options = {}
    probes = {"tts": tts.health, "vlm": vlm.health}
    if guide is None:
        stt = make_stt(models, "stt")
        probes["stt"] = stt.health
        voice_options["stt"] = stt
        capture_backend = _Capture(
            sessions_dir=config.capture.artifacts_dir / "sessions",
            current_frame=frames,
            images=images,
            query_images=MultiImageQueryTool(images=images, vlm=vlm, system_prompt=config.capture.caption_prompt),
            capture_fps=config.capture.capture_fps,
            caption_interval_s=config.capture.caption_interval_s,
        )
        # One owner orders command boundaries and admitted narration, including
        # repeated captures on the same participant connection.
        capture = runtime.register("capture", CaptureAgent(capture_backend))
        endpoint.on_participant(capture.receive)
    else:
        aggregation = runtime.register("voice-aggregation", VoiceAggregationAgent(llm=llm))
        replay = runtime.register("replay", ReplayAgent(
            guide=guide, llm=llm, current_frame=frames, aggregation=aggregation,
            image_query=ImageQueryTool(
                images=images, vlm=vlm,
                system_prompt=(Path(__file__).parent / "replay/prompts/replay_vision.txt").read_text(encoding="utf-8"),
            ),
            vision_timeout_s=config.replay.vision_timeout_s,
        ))
        replay.bind_runtime(runtime)
        probes["llm"] = llm.health
        voice_options = {
            "models": models,
            "participant_joined_topic": PARTICIPANT_JOINED_TOPIC,
            "participant_left_topic": PARTICIPANT_LEFT_TOPIC,
            "interrupted_topic": INTERRUPTED_TOPIC,
            "interrupt_on_supersede": True,
        }
    # Accepted voice input is delivered after prior speech is cancelled, so
    # each capture command cannot interrupt its own acknowledgement.
    # Replay's factory selects and waits for speaker ASR without requiring the
    # batch client retained for capture's ungated narration.
    make_voice = VoiceAgent if capture is not None else VoiceAgent.from_models
    voice = make_voice(
        query_topic=CAPTURE_INPUT_TOPIC if capture is not None else USER_QUERY_TOPIC,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
            # Wait for the final utterance before interpreting capture commands;
            # a partial "stop" must not interrupt narration or acknowledgements.
            stop_probe_after_s=0,
        ),
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes=probes,
        ready_file=ready_file,
        text_input=config.mode == "replay",
        **voice_options,
    )
    runtime.register("voice", voice)
    receiver = asyncio.create_task(endpoint.run())
    try:
        await endpoint.wait_until_running()
        async with runtime:
            await voice.run(runtime)
    finally:
        try:
            # Runtime shutdown drains message delivery, not agent-owned tasks
            # or filesystem threads. Explicitly finalize before closing resources.
            if capture is not None:
                await capture.close()
            if replay is not None:
                await replay.stop()
            if aggregation is not None:
                await aggregation.stop()
        finally:
            # stop() alone cannot wake an idle ZMQ receive. Drain captures
            # first, then cancel the receiver before closing its sockets.
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            endpoint.stop()
            endpoint.close()
            images.clear()
            await vlm.close()
            if llm is not None:
                await llm.close()
