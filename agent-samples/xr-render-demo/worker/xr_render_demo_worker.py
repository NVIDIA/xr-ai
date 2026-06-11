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

from fastmcp import Client as McpClient
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_logging import setup_logging
from xr_ai_models import ToolDef, load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_pipecat import VadConfig, make_voice_pipeline
from xr_ai_pipecat.services import mcp_probe, wait_for_services
from xr_ai_pipecat.transport import XRMediaHubTransport
from xr_ai_voicegate import load_voice_gate_config

from agent import RenderDemoAgent
from config import WorkerConfig, load_config
from processors import RenderSceneProcessor

_TRACE_FILE = "/tmp/xr-agent-trace.log"

# Tools the worker calls directly (control-plane). Excluded from the LLM tool
# list so the model can't trigger them — the worker manages XR lifecycle.
# get_scene_state is intentionally absent: the model must call it to discover
# object ids before any manipulation.
_WORKER_MANAGED_TOOLS = frozenset({"start_xr", "get_health"})


def _build_tools(render_tools: list, oxr_tools: list,
                 vlm_tools: list = (), video_tools: list = (),
                 vec_tools: list = ()) -> list[ToolDef]:
    """Convert MCP tool definitions to ToolDef objects for the SDK."""
    tools: list[ToolDef] = []
    # vec-mcp's pure-math primitives sit next to oxr-mcp's pose-driven helpers
    # in the tool list so the model sees them as a single spatial toolbox.
    all_tools = (list(oxr_tools) + list(vec_tools) + list(render_tools)
                 + list(vlm_tools) + list(video_tools))
    for t in all_tools:
        if t.name in _WORKER_MANAGED_TOOLS:
            continue
        schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        tools.append(ToolDef(
            name=t.name,
            description=(t.description or "").strip(),
            parameters=schema,
        ))
    return tools


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

    models_cfg  = load_models_config(cfg.models_yaml)
    llm         = make_llm(models_cfg, "llm")
    agent_llm   = make_llm(models_cfg, "agent_llm")
    stt         = make_stt(models_cfg, "stt")
    tts         = make_tts(models_cfg, "tts")
    vlm_service = make_vlm(models_cfg, "vlm")

    # VLM /health only returns 200 after weights are fully loaded — this ensures
    # GPU 0 memory has settled before LOVR starts its Vulkan device, preventing
    # the transient OOM race condition.
    probes = {
        "LLM":       llm.health,
        "agent-LLM": agent_llm.health,
        "STT":       stt.health,
        "TTS":       tts.health,
        "VLM":       vlm_service.health,
        "render-mcp":mcp_probe(cfg.render_mcp.rstrip("/") + "/mcp"),
        "oxr-mcp":   mcp_probe(cfg.oxr_mcp.rstrip("/")   + "/mcp"),
        "vlm-mcp":   mcp_probe(cfg.vlm_mcp.rstrip("/")   + "/mcp"),
        "video-mcp": mcp_probe(cfg.video_mcp.rstrip("/") + "/mcp"),
        "vec-mcp":   mcp_probe(cfg.vec_mcp.rstrip("/")   + "/mcp"),
    }
    await wait_for_services(probes)
    await vlm_service.close()

    voice_gate_cfg = load_voice_gate_config(pathlib.Path(cfg.voice_gate_yaml))

    if ready_file:
        ready_file.touch()

    async with (
        McpClient(cfg.render_mcp.rstrip("/") + "/mcp") as render,
        McpClient(cfg.oxr_mcp.rstrip("/")    + "/mcp") as oxr,
        McpClient(cfg.vlm_mcp.rstrip("/")    + "/mcp") as vlm_mcp,
        McpClient(cfg.video_mcp.rstrip("/")  + "/mcp") as video,
        McpClient(cfg.vec_mcp.rstrip("/")    + "/mcp") as vec,
    ):
        render_tools, oxr_tools, vlm_tools, video_tools, vec_tools = [], [], [], [], []
        for name, client, store in [
            ("render-mcp", render,  lambda t: render_tools.extend(t)),
            ("oxr-mcp",    oxr,     lambda t: oxr_tools.extend(t)),
            ("vlm-mcp",    vlm_mcp, lambda t: vlm_tools.extend(t)),
            ("video-mcp",  video,   lambda t: video_tools.extend(t)),
            ("vec-mcp",    vec,     lambda t: vec_tools.extend(t)),
        ]:
            try:
                discovered = await client.list_tools()
                store(discovered)
                logger.info("{} tools: {}", name, [t.name for t in discovered])
            except Exception as exc:
                logger.warning("{} tool discovery failed: {}", name, exc)

        tools = _build_tools(render_tools, oxr_tools, vlm_tools, video_tools, vec_tools)
        logger.info("tool-calling tools: {}", [t.name for t in tools])

        transport = XRMediaHubTransport()
        brain = RenderSceneProcessor(
            transport=transport,
            cfg=cfg,
            render=render,
            oxr=oxr,
            vlm=vlm_mcp,
            video=video,
            vec=vec,
            prompt_path=_PROMPT_FILE,
            tools=tools,
            llm=llm,
            agent_llm=agent_llm,
        )
        # Wire xr.session.started → start_xr lifecycle and the typed-text
        # input path. The agent registers callbacks on the transport's
        # endpoint; those bound methods keep it alive for the worker's
        # lifetime.
        _agent = RenderDemoAgent(transport=transport, brain=brain, render=render)  # noqa: F841

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
    logger.info("xr_render_demo stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    cfg = load_config(ns.config)
    asyncio.run(main(cfg, config_path=ns.config, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
