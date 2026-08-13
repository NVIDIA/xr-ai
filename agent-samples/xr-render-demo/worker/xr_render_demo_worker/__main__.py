# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker with runtime-routed agent input and voice output."""

from __future__ import annotations

import argparse
import asyncio
import pathlib
from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import ChatMessage, load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.tool_calling import tool_definitions
from xr_ai_voice import (
    VadConfig,
    VoiceAgent,
    VoiceSession,
)
from xr_ai_voicegate import load_voice_gate_config

from .agent import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    RenderAgent,
)
from .config import WorkerConfig, load_config
from .lifecycle import XRSessionLifecycle
from .scene_loop import (
    LIVE_PERCEPTION_TOOL,
    PAST_PERCEPTION_TOOL,
    PERCEPTION_TOOL_DEFS,
    SceneModelLoop,
)
from .tools import NativeCapabilities

_TRACE_FILE = "/tmp/xr-agent-trace.log"

_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system.txt"


async def _probe_warmed_llm(llm, *, warmup: bool) -> bool:
    """Report ready only after the optional first-turn warmup succeeds."""

    if not await llm.health():
        return False
    if not warmup:
        return True
    try:
        await llm.chat(
            [ChatMessage(role="user", content="Add a small cube.")],
            max_tokens=40,
            timeout=120.0,
        )
    except Exception:
        logger.opt(exception=True).warning("LLM warmup failed; readiness will retry")
        return False
    return True


async def main(
    cfg: WorkerConfig,
    config_path: pathlib.Path | None = None,
    ready_file: pathlib.Path | None = None,
) -> None:
    setup_logging("worker")

    # Curated session transcript — only records bound with extra={"trace": True}
    # via ``logger.bind(trace=True)`` reach this sink.  Tail this file (or
    # paste it) to see USER/CTX/TOOL/RES/RESP events without the full chatter.
    # DEBUG so verbose CTX / TOOL records (demoted out of the terminal) still
    # land here.
    logger.add(
        _TRACE_FILE,
        filter=lambda r: r["extra"].get("trace") is True,
        format="{time:HH:mm:ss}  {message}",
        mode="w",
        level="DEBUG",
    )
    logger.bind(trace=True).info("=== trace started ===")

    models_cfg = load_models_config(cfg.models_yaml)
    llm = make_llm(models_cfg, "llm")
    agent_llm = make_llm(models_cfg, "agent_llm")
    stt = make_stt(models_cfg, "stt")
    tts = make_tts(models_cfg, "tts")
    vlm_service = make_vlm(models_cfg, "vlm")

    async def llm_probe() -> bool:
        return await _probe_warmed_llm(
            llm,
            warmup=models_cfg.llm("llm").health_check,
        )

    voice_gate_cfg = load_voice_gate_config(pathlib.Path(cfg.voice_gate_yaml))
    session = VoiceSession(
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=cfg.silence_duration,
            min_speech=cfg.min_speech,
            silero_threshold=cfg.silero_threshold,
        ),
        voice_gate=voice_gate_cfg,
        # Readiness, including the LLM warmup, settles model GPU memory before
        # an XR session can ask the scene process to create LOVR's Vulkan device.
        probes={
            "LLM": llm_probe,
            "agent-LLM": agent_llm.health,
            "VLM": vlm_service.health,
        },
        ready_file=ready_file,
        closeables=(),
        # SceneModelLoop publishes the panel response itself.
        text_topic="",
        idle_timeout_secs=cfg.idle_timeout_secs,
    )

    capabilities = NativeCapabilities(
        scene_endpoint=cfg.scene_endpoint,
        openxr_endpoint=cfg.openxr_endpoint,
        video_memory_endpoint=cfg.video_memory_endpoint,
        frame_endpoint=session.transport.endpoint,
        vlm=vlm_service,
        text_memory_dir=cfg.text_memory_dir,
    )
    try:
        # Participant and utterance context are injected by the render agent, so
        # the model sees the established trimmed perception schemas.
        model_tools = [
            tool
            for tool in tool_definitions(capabilities.model)
            if tool.name not in {LIVE_PERCEPTION_TOOL, PAST_PERCEPTION_TOOL}
        ]
        model_tools.extend(PERCEPTION_TOOL_DEFS)
        logger.info("native model tools: {}", [tool.name for tool in model_tools])

        scene_loop = SceneModelLoop(
            transport=session.transport,
            cfg=cfg,
            tools=capabilities.all,
            release_vision=capabilities.release,
            text_memory=capabilities.text_memory,
            prompt_path=_PROMPT_FILE,
            model_tools=model_tools,
            llm=llm,
            agent_llm=agent_llm,
        )
        runtime = AgentRuntime()
        voice = VoiceAgent(
            session,
            query_topic=USER_QUERY_TOPIC,
            text_ignore_topics={"xr.session.started"},
            participant_left_topic=PARTICIPANT_LEFT_TOPIC,
            interrupted_topic=INTERRUPTED_TOPIC,
        )
        runtime.register("voice", voice)
        render = runtime.register("xr-render", RenderAgent(scene_loop))
        # The endpoint retains this bound callback for the worker lifetime.
        XRSessionLifecycle(
            transport=session.transport,
            scene_loop=scene_loop,
            start_xr=capabilities.scene.start_xr,
            get_health=capabilities.scene.get_health,
            runtime=runtime,
        )

        logger.info("xr_render_demo starting")
        async with runtime:
            try:
                await voice.run(runtime)
            finally:
                await render.stop()
    finally:
        await asyncio.gather(
            capabilities.close(),
            session.close(),
            llm.close(),
            agent_llm.close(),
            vlm_service.close(),
            return_exceptions=True,
        )
    logger.info("xr_render_demo stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    cfg = load_config(ns.config)
    asyncio.run(main(cfg, config_path=ns.config, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
