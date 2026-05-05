# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-render-demo agent worker — voice-driven XR scene control via Pipecat.

Pipeline (per participant):
  XRMediaHubInput → SttProcessor(Silero VAD) → RenderSceneProcessor → TtsProcessor → XRMediaHubOutput

Launched as a subprocess by ``uv run xr_render_demo``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import signal
from pathlib import Path

from fastmcp import Client as McpClient

from agent import RenderDemoAgent
from config import WorkerConfig, load_config
from processors import _setup_trace_log
from xr_ai_pipecat.services import http_probe, mcp_probe, wait_for_services

log = logging.getLogger("xr_render_demo")

# Tools the worker calls directly (control-plane). Excluded from the LLM tool
# list so the model can't trigger them — the worker manages XR lifecycle.
# get_scene_state is intentionally absent: the model must call it to discover
# object ids before any manipulation.
_WORKER_MANAGED_TOOLS = frozenset({"start_xr", "get_health"})


def _tool_param_sig(input_schema: dict) -> str:
    props = (input_schema or {}).get("properties", {})
    if not props:
        return ""
    return ", ".join(
        f"{k}: {v.get('type', 'any')}" for k, v in props.items()
    )


def _build_tools_openai(render_tools: list, oxr_tools: list,
                        vlm_tools: list = (), video_tools: list = ()) -> list:
    """Convert MCP tool definitions to the OpenAI tools=[...] format."""
    tools = []
    all_tools = list(oxr_tools) + list(render_tools) + list(vlm_tools) + list(video_tools)
    for t in all_tools:
        if t.name in _WORKER_MANAGED_TOOLS:
            continue
        schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        tools.append({
            "type": "function",
            "function": {
                "name":        t.name,
                "description": (t.description or "").strip(),
                "parameters":  schema,
            },
        })
    return tools


_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system.txt"




async def main(cfg: WorkerConfig, ready_file: pathlib.Path | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _setup_trace_log("/tmp/xr-agent-trace.log")

    await wait_for_services({
        "STT":       http_probe(cfg.stt_server.rstrip("/")       + "/health"),
        "TTS":       http_probe(cfg.tts_server.rstrip("/")       + "/health"),
        "LLM":       http_probe(cfg.llm_server.rstrip("/")       + "/health"),
        "agent-LLM": http_probe(cfg.agent_llm_server.rstrip("/") + "/health"),
        # VLM server /health only returns 200 AFTER model weights are fully
        # loaded — this ensures GPU 0 memory has settled before LOVR starts
        # its Vulkan device, preventing the transient OOM race condition.
        "VLM":       http_probe(cfg.vlm_server.rstrip("/")       + "/health"),
        "render-mcp":mcp_probe(cfg.render_mcp.rstrip("/")  + "/mcp"),
        "oxr-mcp":   mcp_probe(cfg.oxr_mcp.rstrip("/")    + "/mcp"),
        "vlm-mcp":   mcp_probe(cfg.vlm_mcp.rstrip("/")    + "/mcp"),
        "video-mcp": mcp_probe(cfg.video_mcp.rstrip("/")  + "/mcp"),
    })

    if ready_file:
        ready_file.touch()

    async with (
        McpClient(cfg.render_mcp.rstrip("/") + "/mcp") as render,
        McpClient(cfg.oxr_mcp.rstrip("/")    + "/mcp") as oxr,
        McpClient(cfg.vlm_mcp.rstrip("/")    + "/mcp") as vlm,
        McpClient(cfg.video_mcp.rstrip("/")  + "/mcp") as video,
    ):
        render_tools, oxr_tools, vlm_tools, video_tools = [], [], [], []
        for name, client, store in [
            ("render-mcp", render, lambda t: render_tools.extend(t)),
            ("oxr-mcp",    oxr,    lambda t: oxr_tools.extend(t)),
            ("vlm-mcp",    vlm,    lambda t: vlm_tools.extend(t)),
            ("video-mcp",  video,  lambda t: video_tools.extend(t)),
        ]:
            try:
                tools = await client.list_tools()
                store(tools)
                log.info("%s tools: %s", name, [t.name for t in tools])
            except Exception as exc:
                log.warning("%s tool discovery failed: %s", name, exc)

        tools_openai = _build_tools_openai(render_tools, oxr_tools, vlm_tools, video_tools)
        log.info("tool-calling tools: %s", [t["function"]["name"] for t in tools_openai])

        agent = RenderDemoAgent(
            cfg, render, oxr, vlm, video, _PROMPT_FILE, tools_openai
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, agent.shutdown)

        log.info("xr_render_demo starting")
        try:
            await agent.run()
        finally:
            agent.shutdown()
    log.info("xr_render_demo stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    cfg = load_config(ns.config)
    asyncio.run(main(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
