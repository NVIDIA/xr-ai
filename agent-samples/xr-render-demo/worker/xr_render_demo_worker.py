# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-render-demo agent worker — voice-driven XR scene control via Pipecat.

Voice pipeline (assembled by ``xr_ai_pipecat.make_voice_pipeline``):
  transport.input → VadStt → VoiceGate → RenderSceneProcessor (brain)
                  → StreamingTts → transport.output

Launched as a subprocess by ``uv run xr_render_demo``.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal
from pathlib import Path

from loguru import logger
from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionGroupRef, LLMRef
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_nat.functions.text_memory import TextMemoryFunctionsConfig
from xr_ai_nat.functions.vision import LiveVisionFunctionConfig
from xr_ai_nat.llm import ModelsLLMConfig
from xr_ai_pipecat import VadConfig, make_voice_pipeline
from xr_ai_pipecat.services import wait_for_services
from xr_ai_pipecat.transport import XRMediaHubTransport
from xr_ai_voicegate import load_voice_gate_config

from agent import RenderDemoAgent
from capabilities import build_native_toolbox
from config import WorkerConfig, load_config
from processors import _PERCEPTION_SYSTEM_PROMPT, _PERCEPTION_TOOL_DEF, RenderSceneProcessor

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
_VALIDATOR_PROMPT_FILE = _PROMPT_FILE.parent / "validate.txt"


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

    # VLM /health only returns 200 after weights are fully loaded — this ensures
    # GPU 0 memory has settled before LOVR starts its Vulkan device, preventing
    # the transient OOM race condition.
    probes = {
        "LLM": llm.health,
        "agent-LLM": agent_llm.health,
        "STT": stt.health,
        "TTS": tts.health,
        "VLM": vlm_service.health,
    }
    await wait_for_services(probes)

    voice_gate_cfg = load_voice_gate_config(pathlib.Path(cfg.voice_gate_yaml))

    transport = XRMediaHubTransport()
    live_vision_config = LiveVisionFunctionConfig(
        endpoint=transport.endpoint,
        vlm=vlm_service,
        system_prompt=_PERCEPTION_SYSTEM_PROMPT,
    )
    async with WorkflowBuilder() as builder:
        toolbox = await build_native_toolbox(
            builder,
            scene_endpoint=cfg.scene_endpoint,
            openxr_endpoint=cfg.openxr_endpoint,
            video_memory_endpoint=cfg.video_memory_endpoint,
            vlm=vlm_service,
        )
        await builder.add_function_group(
            "text_memory", TextMemoryFunctionsConfig(directory=cfg.text_memory_dir)
        )

        live_vision = await builder.add_function("live_vision", live_vision_config)
        text_memory_functions = await _group_functions(builder, "text_memory")
        text_memory = text_memory_functions["text_memory__add_transcript"]
        validator_llm = LLMRef("scene_validator_llm")
        await builder.add_llm(
            validator_llm,
            ModelsLLMConfig(
                service=llm,
                model_name="xr-scene-validator",
                max_tokens=60,
                temperature=0.0,
            ),
        )
        validator = await builder.add_function(
            "scene_validator",
            ToolCallAgentWorkflowConfig(
                llm_name=validator_llm,
                tool_names=[FunctionGroupRef("scene_state")],
                system_prompt=_VALIDATOR_PROMPT_FILE.read_text(encoding="utf-8").strip(),
                description="Check whether an XR scene request was completed.",
                max_iterations=1,
            ),
        )
        tools = toolbox.definitions(exclude=_WORKER_MANAGED_TOOLS)
        tools.append(_PERCEPTION_TOOL_DEF)
        logger.info("native tool-calling functions: {}", [tool.name for tool in tools])

        brain = RenderSceneProcessor(
            transport=transport,
            cfg=cfg,
            toolbox=toolbox,
            live_vision=live_vision,
            release_vision=live_vision_config.release,
            text_memory=text_memory,
            validator=validator,
            prompt_path=_PROMPT_FILE,
            tools=tools,
            llm=llm,
            agent_llm=agent_llm,
        )
        # Wire xr.session.started → start_xr lifecycle and the typed-text
        # input path. The agent registers callbacks on the transport's
        # endpoint; those bound methods keep it alive for the worker's
        # lifetime.
        _agent = RenderDemoAgent(transport=transport, brain=brain, tools=toolbox)  # noqa: F841

        if ready_file:
            ready_file.touch()

        _, task = make_voice_pipeline(
            transport=transport,
            stt=stt,
            tts=tts,
            brain=brain,
            vad_cfg=VadConfig(
                silence_duration=cfg.silence_duration,
                min_speech=cfg.min_speech,
                silero_threshold=cfg.silero_threshold,
            ),
            voice_gate_cfg=voice_gate_cfg,
            # Brain pushes its own per-turn ``agent.response`` data
            # message with the sanitized "display" string (see
            # ``RenderDemoBrain._run_turn``); opting out of the
            # pipeline-level echo here avoids a duplicate send.
            text_topic="",
            # Idle-timeout auto-cancel — disabled unless set in the worker YAML.
            idle_timeout_secs=cfg.idle_timeout_secs,
        )

        loop = asyncio.get_running_loop()
        cancel_requested = False

        def _request_cancel() -> None:
            # PipelineTask.cancel is a coroutine; add_signal_handler needs a
            # sync callable. Guard against a second signal (e.g. double
            # ctrl-c) spawning a redundant cancel task while the first is
            # still draining the pipeline.
            nonlocal cancel_requested
            if cancel_requested:
                return
            cancel_requested = True
            asyncio.create_task(task.cancel())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _request_cancel)

        logger.info("xr_render_demo starting")
        try:
            await PipelineRunner().run(task)
        finally:
            transport.shutdown()
            await brain.close()
            for service in (stt, tts, vlm_service):
                try:
                    await service.close()
                except Exception:
                    logger.opt(exception=True).warning("service close failed")
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
