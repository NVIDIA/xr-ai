# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Silent narration input and camera-controlled multimodal capture."""

from __future__ import annotations

import asyncio
from pathlib import Path

from xr_ai_hub import ProcessorEndpoint
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import UserQuery, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from .config import WorkerConfig
from .lifecycle import CameraRecording
from .recorder import RecorderAgent

_HUB_PUB = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker")
    models = load_models_config(config.models_config)
    stt, tts, vlm = make_stt(models, "stt"), make_tts(models, "tts"), make_vlm(models, "vlm")
    # Capture control has its own endpoint so VoiceAgent can own its transport
    # lifecycle without closing the path needed to finalize pending recordings.
    endpoint = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=endpoint, images=images, frame_max_age_s=config.frame_max_age_s, frame_timeout_s=config.frame_timeout_s
    )
    recorder = RecorderAgent(
        sessions_dir=config.artifacts_dir / "sessions",
        capture_endpoint=endpoint,
        media_capture_dir=config.media_capture_dir,
        current_frame=frames,
        images=images,
        query_image=ImageQueryTool(images=images, vlm=vlm, system_prompt=config.caption_prompt),
        capture_fps=config.capture_fps,
        caption_interval_s=config.caption_interval_s,
    )
    camera = CameraRecording(recorder)
    endpoint.on_participant(camera.receive)
    endpoint.on_video_track(camera.receive)
    voice = VoiceAgent(
        query_topic=Topic("sop.narration", UserQuery),
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
            # Passive narration has no output to interrupt or wake acknowledgement.
            stop_probe_after_s=0,
        ),
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes={"stt": stt.health, "vlm": vlm.health},
        ready_file=ready_file,
        text_input=False,
    )
    runtime = AgentRuntime()
    runtime.register("voice", voice)
    # No query subscriber or voice output producer: all speech is narration.
    receiver = asyncio.create_task(endpoint.run())
    try:
        await endpoint.wait_until_running()
        async with runtime:
            await voice.run(runtime)
    finally:
        try:
            await camera.close()
        finally:
            # stop() alone cannot wake an idle ZMQ receive. Drain recordings
            # first, then cancel the receiver before closing its sockets.
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            endpoint.stop()
            endpoint.close()
            await vlm.close()
