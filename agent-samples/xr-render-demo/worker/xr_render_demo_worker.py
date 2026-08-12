# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-render-demo worker with runtime-routed agent input and voice output."""

from __future__ import annotations

import argparse
import asyncio
import pathlib
from pathlib import Path

from loguru import logger
from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_logging import setup_logging
from xr_ai_models import ChatMessage, load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_nat.functions.text_memory import TextMemoryFunctionsConfig
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import (
    VadConfig,
    VoiceAgent,
    VoiceSession,
)
from xr_ai_voicegate import load_voice_gate_config

from agent import RenderDemoAgent
from capabilities import build_native_toolbox
from config import WorkerConfig, load_config
from dispatch import (
    CancelAllRender,
    CancelRender,
    RenderAgent,
    USER_QUERY_TOPIC,
)
from processors import (
    _LIVE_PERCEPTION_TOOL,
    _PAST_PERCEPTION_TOOL,
    _PERCEPTION_TOOL_DEFS,
    RenderSceneAgent,
)

_TRACE_FILE = "/tmp/xr-agent-trace.log"

# Tools the worker calls directly (control-plane). Excluded from the LLM tool
# list so the model can't trigger them — the worker manages XR lifecycle.
# get_scene_state is intentionally absent: the model must call it to discover
# object ids before any manipulation.
_WORKER_MANAGED_TOOLS = frozenset({"start_xr", "get_health"})


async def _group_functions(builder: WorkflowBuilder, *names: str) -> dict[str, Function]:
    functions: dict[str, Function] = {}
    for name in names:
        group = await builder.get_function_group(name)
        functions.update(await group.get_all_functions())
    return functions


_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system.txt"


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
        probes={
            "LLM": llm.health,
            "agent-LLM": agent_llm.health,
            "VLM": vlm_service.health,
        },
        ready_file=ready_file,
        closeables=(llm, agent_llm, vlm_service),
        text_topic="",
        idle_timeout_secs=cfg.idle_timeout_secs,
    )

    async with WorkflowBuilder() as builder:
        # VLM readiness must settle GPU memory before LOVR creates its Vulkan device.
        if models_cfg.llm("llm").health_check:
            try:
                await llm.chat(
                    [ChatMessage(role="user", content="Add a small cube.")],
                    max_tokens=40,
                    timeout=120.0,
                )
            except Exception:
                logger.opt(exception=True).warning("LLM warmup failed")

        toolbox, vision_config = await build_native_toolbox(
            builder,
            scene_endpoint=cfg.scene_endpoint,
            openxr_endpoint=cfg.openxr_endpoint,
            video_memory_endpoint=cfg.video_memory_endpoint,
            frame_endpoint=session.transport.endpoint,
            vlm=vlm_service,
        )
        await builder.add_function_group(
            "text_memory", TextMemoryFunctionsConfig(directory=cfg.text_memory_dir)
        )

        text_memory_functions = await _group_functions(builder, "text_memory")
        text_memory = text_memory_functions["text_memory__add_transcript"]
        # The native perception request models carry participant/reference
        # context the processor injects; present the model trimmed schemas
        # (question, and question+second_ago) in place of the raw native ones.
        tools = toolbox.definitions(
            exclude=_WORKER_MANAGED_TOOLS | {_LIVE_PERCEPTION_TOOL, _PAST_PERCEPTION_TOOL}
        )
        tools.extend(_PERCEPTION_TOOL_DEFS)
        logger.info("native tool-calling functions: {}", [tool.name for tool in tools])

        scene_agent = RenderSceneAgent(
            transport=session.transport,
            cfg=cfg,
            toolbox=toolbox,
            release_vision=vision_config.release,
            text_memory=text_memory,
            prompt_path=_PROMPT_FILE,
            tools=tools,
            llm=llm,
            agent_llm=agent_llm,
        )
        runtime = AgentRuntime()
        render_ref = runtime.register("xr-render", RenderAgent(scene_agent))
        RenderDemoAgent(
            transport=session.transport,
            scene_agent=scene_agent,
            tools=toolbox,
            runtime=runtime,
        )
        async def participant_left(participant_id: str) -> None:
            await runtime.call(
                render_ref,
                CancelRender(),
                participant_id=participant_id,
                source="voice.participant-left",
            )
            await scene_agent.on_participant_left(participant_id)

        async def interrupted(participant_id: str | None) -> None:
            request = CancelRender() if participant_id is not None else CancelAllRender()
            await runtime.call(
                render_ref,
                request,
                participant_id=participant_id or "voice-runtime",
                source="voice.interruption",
            )

        voice = VoiceAgent(
            session,
            query_topic=USER_QUERY_TOPIC,
            text_ignore_topics={"xr.session.started"},
            on_participant_left=participant_left,
            on_interrupted=interrupted,
            interrupt_on_supersede=True,
        )
        runtime.register("voice", voice)

        logger.info("xr_render_demo starting")
        async with runtime:
            await voice.wait()
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
