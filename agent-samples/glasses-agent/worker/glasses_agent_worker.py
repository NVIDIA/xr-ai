# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
glasses-agent worker — always-on AI assistant for smart glasses.

Pipeline:
  Hub audio → VadDetector (per participant) → STT → QueryProcessor
                                                          │
                                              ┌───────────┴────────────────┐
                                              │  demo detection / guidance │
                                              │  quick-ack + agentic loop  │
                                              └───────────┬────────────────┘
                                                    TTS audio + data
  Background: VLM loop  →  AgentMemory  →  TranscriptClient → transcript-mcp
              Condenser  →  scene summary

Launch:
  cd glasses-agent/worker && uv run glasses_agent_worker
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import signal
import time

import yaml
from fastmcp import Client as McpClient
from xr_ai_logging import setup_logging

from agent import GlassesAgent
from config import WorkerConfig, load_config
from memory import AgentMemory, Observation, Demonstration, DemoStep, TranscriptClient
from processors import QueryProcessor

log = logging.getLogger("glasses_agent")

# ── trace log ─────────────────────────────────────────────────────────────────

_trace_log = logging.getLogger("glasses_agent.trace")


def _setup_trace_log(path: str = "/tmp/glasses-agent-trace.log") -> None:
    h = logging.FileHandler(path, mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    _trace_log.addHandler(h)
    _trace_log.setLevel(logging.DEBUG)
    _trace_log.propagate = False
    _trace_log.info("=== glasses-agent trace started ===")


# ── service probes ────────────────────────────────────────────────────────────

async def _http_probe(url: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(url)
            return r.is_success
    except Exception:
        return False


async def _mcp_probe(mcp_url: str) -> bool:
    """Check that the FastMCP endpoint is serving tools via StreamableHTTP."""
    from fastmcp import Client as McpClient
    try:
        async with McpClient(mcp_url) as client:
            await client.list_tools()
        return True
    except Exception:
        return False


async def _wait_for_all(probes_spec: dict[str, tuple]) -> None:
    """Wait for all services. probes_spec: {name: ("http"|"mcp", url)}"""
    pending = set(probes_spec)
    while pending:
        done: list[str] = []
        for name in list(pending):
            kind, url = probes_spec[name]
            if kind == "http":
                ok = await _http_probe(url)
            else:
                ok = await _mcp_probe(url)
            if ok:
                log.info("%s ready", name)
                done.append(name)
        pending -= set(done)
        if pending:
            log.info("still waiting for: %s", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)


# ── tool discovery ────────────────────────────────────────────────────────────

def _build_tools_openai(vlm_tools: list, video_tools: list) -> list:
    """Convert MCP tool definitions to OpenAI tools=[...] format."""
    tools = []
    for t in list(vlm_tools) + list(video_tools):
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


# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an AI assistant integrated into smart glasses. You observe the world \
through the wearer's camera and help them understand their environment, \
remember past events, and learn from demonstrations.

CONTEXT PROVIDED EACH TURN:
- Scene summary: condensed description of recent observations
- Recent observations: timestamped timeline of what you've seen
- Available demonstrations: labeled procedures that were recorded
- Latest camera frame path (when available — use ask_image if you need to examine it)
- Conversation history (last 4 turns)

CAPABILITIES:
1. OBSERVE: Describe what the wearer is currently seeing using the latest frame
2. REMEMBER: Answer questions about past observations from the timeline
3. GUIDE: Walk the wearer through a stored demonstration step by step
4. ANALYZE: Use ask_image to examine the current view or a past frame closely
5. INVESTIGATE: Use get_frame_from_time to look at specific past moments

BEHAVIOR:
- Be concise — the wearer is doing things while listening.
- Respond with 1-3 short sentences maximum unless giving step-by-step guidance.
- Use the observations timeline to answer "what happened" questions before \
fetching past frames with tools.
- When asked to recall a demonstration, describe each step clearly.
- If you're unsure about something in the current view, use ask_image with \
the pre-fetched frame path from context.
- Reference time (when user spoke) is provided so past-frame lookups are \
anchored correctly — always pass it as reference_time_us.
- Never invent or hallucinate observations — only report what you've actually seen.
- CRITICAL: Never construct, guess, or fabricate an image file path. The ONLY \
valid image_path for ask_image is one returned directly by get_latest_frame or \
get_frame_from_time in the current turn. If no frame path is in context, call \
get_latest_frame first.
"""


# ── startup memory restore ─────────────────────────────────────────────────────

async def _restore_memory(
    memory: AgentMemory,
    transcript: TranscriptClient,
    source: str,
    window_min: int = 20,
) -> None:
    """Load recent observations and demonstrations from transcript-mcp into AgentMemory."""
    window_us = window_min * 60 * 1_000_000
    log.info("restoring memory from transcript-mcp  window=%d min", window_min)

    # Restore observations.
    obs_entries = await transcript.query_recent(source + ":observations", window_us)
    restored = 0
    for entry in obs_entries:
        ts  = entry.get("timestamp_us", 0)
        txt = entry.get("text", "").strip()
        if txt:
            memory.restore_observation(Observation(
                timestamp_us = ts,
                description  = txt,
                image_path   = "",  # image may be gone from /tmp; description is what matters
            ))
            restored += 1
    log.info("restored %d observations", restored)

    # Restore demonstrations — stored as JSON blobs in the "demonstrations" source.
    demo_entries = await transcript.query_recent(source + ":demonstrations", window_us * 24 * 7)
    for entry in demo_entries:
        txt = entry.get("text", "").strip()
        if not txt:
            continue
        try:
            import json
            obj = json.loads(txt)
            demo = Demonstration(
                name          = obj["name"],
                started_at_us = obj.get("started_at_us", 0),
                ended_at_us   = obj.get("ended_at_us",   0),
                summary       = obj.get("summary", ""),
                steps=[
                    DemoStep(
                        step_number  = s["step_number"],
                        timestamp_us = s.get("timestamp_us", 0),
                        description  = s["description"],
                        image_path   = s.get("image_path", ""),
                    )
                    for s in obj.get("steps", [])
                ],
            )
            memory.restore_demonstration(demo)
            log.info("restored demonstration %r (%d steps)", demo.name, len(demo.steps))
        except Exception as exc:
            log.warning("failed to restore demo entry: %s", exc)


async def _persist_demo(
    demo: Demonstration,
    transcript: TranscriptClient,
    source: str,
) -> None:
    """Persist a finished demonstration as a JSON blob in transcript-mcp."""
    import json as _json
    obj = {
        "name":          demo.name,
        "started_at_us": demo.started_at_us,
        "ended_at_us":   demo.ended_at_us,
        "summary":       demo.summary,
        "steps": [
            {
                "step_number":  s.step_number,
                "timestamp_us": s.timestamp_us,
                "description":  s.description,
                "image_path":   s.image_path,
            }
            for s in demo.steps
        ],
    }
    await transcript.add_entry(
        source + ":demonstrations",
        demo.ended_at_us or int(time.time() * 1_000_000),
        _json.dumps(obj),
    )
    log.info("persisted demonstration %r", demo.name)


# ── main ──────────────────────────────────────────────────────────────────────

async def main(cfg: WorkerConfig, ready_file: pathlib.Path | None = None) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    _run_dir = os.environ.get("XR_RUN_DIR")
    _trace_path = str(pathlib.Path(_run_dir) / "trace.log") if _run_dir else "/tmp/glasses-agent-trace.log"
    _setup_trace_log(_trace_path)

    log.info("glasses-agent starting")

    # ── wait for all services ─────────────────────────────────────────────────
    await _wait_for_all({
        "STT":          ("http", cfg.stt_server.rstrip("/")        + "/health"),
        "TTS":          ("http", cfg.tts_server.rstrip("/")        + "/health"),
        "LLM":          ("http", cfg.llm_server.rstrip("/")        + "/health"),
        "agent-LLM":    ("http", cfg.agent_llm_server.rstrip("/")  + "/health"),
        "vlm-mcp":      ("mcp",  cfg.vlm_mcp.rstrip("/")           + "/mcp"),
        "video-mcp":    ("mcp",  cfg.video_mcp.rstrip("/")         + "/mcp"),
        "transcript-mcp":("mcp", cfg.transcript_mcp.rstrip("/")    + "/mcp"),
    })

    if ready_file:
        ready_file.touch()

    # ── connect to MCP servers + discover tools ───────────────────────────────
    async with (
        McpClient(cfg.vlm_mcp.rstrip("/")       + "/mcp") as vlm_mcp,
        McpClient(cfg.video_mcp.rstrip("/")     + "/mcp") as video_mcp,
    ):
        vlm_tools, video_tools = [], []
        for name, client, store in [
            ("vlm-mcp",   vlm_mcp,   lambda t: vlm_tools.extend(t)),
            ("video-mcp", video_mcp, lambda t: video_tools.extend(t)),
        ]:
            try:
                tools = await client.list_tools()
                store(tools)
                log.info("%s tools: %s", name, [t.name for t in tools])
            except Exception as exc:
                log.warning("%s tool discovery failed: %s", name, exc)

        tools_openai = _build_tools_openai(vlm_tools, video_tools)
        log.info("tool-calling tools: %s", [t["function"]["name"] for t in tools_openai])

        # ── memory + transcript ───────────────────────────────────────────────
        memory     = AgentMemory(max_obs=cfg.vlm_obs_max)
        transcript = TranscriptClient(cfg.transcript_mcp)

        # Restore from transcript-mcp.
        await _restore_memory(memory, transcript, cfg.transcript_source)

        # ── query processor ───────────────────────────────────────────────────

        async def _say(pid: str, text: str) -> None:
            await agent.say(pid, text)

        async def _send_text(pid: str, text: str, topic: str) -> None:
            await agent.send_text(pid, text, topic)

        query_proc = QueryProcessor(
            cfg           = cfg,
            memory        = memory,
            vlm_client    = vlm_mcp,
            video_client  = video_mcp,
            system_prompt = SYSTEM_PROMPT,
            tools_openai  = tools_openai,
            send_text     = _send_text,
            say           = _say,
        )

        # ── main agent ────────────────────────────────────────────────────────
        agent = GlassesAgent(
            cfg               = cfg,
            memory            = memory,
            transcript_client = transcript,
            query_processor   = query_proc,
            stt_url           = cfg.stt_server,
            tts_url           = cfg.tts_server,
            vlm_mcp_url       = cfg.vlm_mcp,
            video_mcp_url     = cfg.video_mcp,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, agent.shutdown)

        log.info("glasses-agent running")
        try:
            await agent.run()
        finally:
            agent.shutdown()
            await query_proc.close()
            await transcript.close()

    log.info("glasses-agent stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    setup_logging("worker", namespace="glasses-agent")
    cfg = load_config(ns.config)
    asyncio.run(main(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
