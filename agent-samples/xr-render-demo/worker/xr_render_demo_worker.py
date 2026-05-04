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


def _build_prompt(render_tools: list, oxr_tools: list,
                  vlm_tools: list = (), video_tools: list = ()) -> str:
    """System prompt for the Llama-Nemotron tool-calling agent.

    LMFE guarantees valid tool calls, so no calibration examples or JSON
    format instructions are needed — just clear task guidance.
    """
    return (
        "You are an AI assistant controlling a 3D XR scene for a user wearing a headset.\n"
        "\n"
        "CONTEXT: Every turn includes '[Pre-fetched context]' with the current scene and "
        "head pose. Use those values directly — do NOT call get_scene_state or get_head_pose "
        "at the start of a turn. Only re-call them after you make changes that change what "
        "you need to read next.\n"
        "\n"
        "═══ PRONOUN & REFERENCE RESOLUTION ═══\n"
        "Before anything else, identify WHICH OBJECT the user means:\n"
        "  'it' / 'that' / 'the one' → the most recently added or modified object "
        "(check conversation history, then scene order).\n"
        "  'the red one' / 'the big sphere' → match by color or size in the scene context.\n"
        "  'sphere one' / 'first sphere' → sphere-1; 'second' → sphere-2, etc.\n"
        "  'all of them' / 'everything' → iterate over every id in the scene.\n"
        "Never guess an id. If genuinely ambiguous, pick the most recently touched object.\n"
        "\n"
        "═══ SPATIAL RELATIONSHIPS ═══\n"
        "COORDINATE SYSTEM: world-space metres, OpenXR Y-up.  "
        "+Y = up, world axes do NOT align with the user's view when the head is rotated.\n"
        "\n"
        "Relative to the USER (use head pose vectors from context):\n"
        "  'in front of me' / 'ahead'    → position_ahead(1.5)\n"
        "  'd metres in front of me'     → position_ahead(d)\n"
        "  'to my right d m'             → position_relative(right=d)\n"
        "  'to my left d m'              → position_relative(right=-d)\n"
        "  'above me' / 'up d m'         → position_relative(up=d)\n"
        "  'behind me d m'               → position_relative(forward=-d)\n"
        "\n"
        "Relative to an EXISTING OBJECT (objects have no orientation — use world axes):\n"
        "  'in front of obj'  → obj.z - d   (world -Z is 'forward')\n"
        "  'behind obj'       → obj.z + d\n"
        "  'right of obj'     → obj.x + d\n"
        "  'left of obj'      → obj.x - d\n"
        "  'above obj'        → obj.y + d\n"
        "  'below/under obj'  → obj.y - d\n"
        "  'next to obj'      → obj.x + d  (default right)\n"
        "  'between obj-A and obj-B' → midpoint: x=(Ax+Bx)/2  y=(Ay+By)/2  z=(Az+Bz)/2\n"
        "\n"
        "MOVING an existing object in a user-relative direction:\n"
        "  Read head.right / head.forward / head.up vectors from context.\n"
        "  new = old + direction_vec × distance   (apply to all three components)\n"
        "  Example: sphere at (1.09, 2.63, -0.00), head.right=(0.00, 0.00, 1.00), move 1 m right:\n"
        "    x = 1.09 + 0.00×1 = 1.09\n"
        "    y = 2.63 + 0.00×1 = 2.63\n"
        "    z = -0.00 + 1.00×1 = 1.00\n"
        "    → update_primitive(id=…, x=1.09, y=2.63, z=1.00)\n"
        "  'left' = -right  |  'back' = -forward  |  'down' = -up\n"
        "\n"
        "═══ REAL-WORLD VISUAL QUERIES ═══\n"
        "If the user references anything about the real world that you cannot determine "
        "from the scene state alone — what they are holding, pointing at, or looking at; "
        "a real-world color, shape, or object — you MUST:\n"
        "  1. Call get_frame_from_time first to capture a frame.\n"
        "  2. Pass the returned path to ask_image with a specific question.\n"
        "  3. Only then take action based on the answer.\n"
        "NEVER guess or assume a real-world color or object. "
        "If you are not certain from the scene context alone, use the visual tools.\n"
        "\n"
        "═══ RULES ═══\n"
        "1. Never invent or guess an object id — only use ids from the scene context.\n"
        "2. add_primitive always creates a NEW object, even if similar ones exist.\n"
        "3. COLOR — always set all three of r, g, b:\n"
        "   red=(1,0,0)  green=(0,0.8,0)  blue=(0,0.4,1)  yellow=(1,1,0)\n"
        "   orange=(1,0.5,0)  purple=(0.6,0,1)  white=(1,1,1)  black=(0,0,0)\n"
        "4. SIZE in metres — radius for spheres, half-edge for boxes:\n"
        "   tiny=0.05  small=0.08  default=0.1  medium=0.2  large=0.5  huge=1.0\n"
        "   Resize: new_size = current_size × factor  (e.g. '3× bigger' → ×3, 'half' → ×0.5)\n"
        "5. After completing the task respond with ONE short sentence confirming what was done.\n"
    )


async def main(cfg: WorkerConfig) -> None:
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

        actions_prompt = _build_prompt(render_tools, oxr_tools, vlm_tools, video_tools)
        tools_openai   = _build_tools_openai(render_tools, oxr_tools, vlm_tools, video_tools)
        log.info("tool-calling tools: %s", [t["function"]["name"] for t in tools_openai])

        agent = RenderDemoAgent(
            cfg, render, oxr, vlm, video, actions_prompt, tools_openai
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
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    cfg = load_config(ns.config)
    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
