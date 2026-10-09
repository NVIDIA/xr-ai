# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose replay voice and verification without capture services or writers."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent, VoiceAggregationAgent
from xr_ai_voicegate import load_voice_gate_config

from .guides import select_guide
from .replay import ReplayAgent
from .replay_events import INTERRUPTED_TOPIC, PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, USER_QUERY_TOPIC


async def run_replay(settings_path: Path, guide_name: str, *, ready_file: Path | None = None) -> None:
    setup_logging("worker")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    base = settings_path.resolve().parent
    # Reject draft, ambiguous, and invalid guides before opening model or hub clients.
    guide = select_guide(base / settings["guides_dir"], guide_name)
    models = load_models_config(base / settings["models_config"])
    llm, vlm = make_llm(models, "llm"), make_vlm(models, "vlm")
    stt, tts = make_stt(models, "stt"), make_tts(models, "tts")
    images = ImageRegistry()
    transport = HubVoiceTransport()
    frames = CurrentFrameTool(
        endpoint=transport.endpoint, images=images,
        frame_max_age_s=float(settings["frame_max_age_s"]),
        frame_timeout_s=float(settings["frame_timeout_s"]),
    )
    voice = VoiceAgent(
        query_topic=USER_QUERY_TOPIC, stt=stt, tts=tts,
        vad=VadConfig(stop_probe_after_s=0),
        voice_gate=load_voice_gate_config(base / settings["voice_gate_yaml"]),
        transport=transport, ready_file=ready_file,
        probes={"llm": llm.health, "vlm": vlm.health, "stt": stt.health, "tts": tts.health},
        closeables=(llm, vlm), text_topic="sop.replay.status",
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC, interrupt_on_supersede=True,
    )
    runtime = AgentRuntime()
    aggregation = runtime.register("voice-aggregation", VoiceAggregationAgent(llm=llm))
    replay = runtime.register("replay", ReplayAgent(
        guide=guide, llm=llm, current_frame=frames, aggregation=aggregation,
        image_query=ImageQueryTool(
            images=images, vlm=vlm,
            system_prompt=(Path(__file__).parent / "prompts/replay_vision.txt").read_text(encoding="utf-8"),
        ),
        vision_timeout_s=float(settings["vision_timeout_s"]),
    ))
    runtime.register("voice", voice)
    replay.bind_runtime(runtime)
    logger.info("Replay guide={} version={} sha256={}", guide.workflow.name, guide.workflow.version, guide.sha256)
    async with runtime:
        try:
            await voice.run(runtime)
        finally:
            await replay.stop()
            await aggregation.stop()
            images.clear()


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay one approved SOP guide.")
    parser.add_argument("--config", type=Path, required=True, help="launcher-owned JSON with settings and guide_name")
    parser.add_argument("--ready-file", type=Path, default=None)
    args = parser.parse_args(argv)
    request = json.loads(args.config.read_text(encoding="utf-8"))
    asyncio.run(run_replay(Path(request["settings"]), request["guide_name"], ready_file=args.ready_file))
