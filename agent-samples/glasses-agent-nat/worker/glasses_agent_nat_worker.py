# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
glasses-agent-nat worker — always-on AI assistant for smart glasses.

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
  cd glasses-agent-nat/worker && uv run glasses_agent_nat_worker
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import signal
import time

from xr_ai_logging import setup_logging

from agent import GlassesAgent
from config import WorkerConfig, load_config
from memory import AgentMemory, Observation, Demonstration, DemoStep, TranscriptClient
from nat_runtime import NatRuntime
from processors import QueryProcessor

log = logging.getLogger("glasses_agent_nat")

# ── trace log ─────────────────────────────────────────────────────────────────

_trace_log = logging.getLogger("glasses_agent_nat.trace")


def _setup_trace_log(path: str = "/tmp/glasses-agent-nat-trace.log") -> None:
    h = logging.FileHandler(path, mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    _trace_log.addHandler(h)
    _trace_log.setLevel(logging.DEBUG)
    _trace_log.propagate = False
    _trace_log.info("=== glasses-agent-nat trace started ===")


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
    """Check that the MCP HTTP endpoint is accepting connections."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as c:
            r = await c.get(mcp_url)
            return r.status_code < 500 and r.status_code != 404
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

# ── startup memory restore ─────────────────────────────────────────────────────

async def _restore_memory(
    memory: AgentMemory,
    transcript: TranscriptClient,
    source: str,
    window_min: int = 20,
) -> None:
    """Load recent observations from transcript-mcp into AgentMemory.

    Demonstrations are intentionally NOT restored across runs — each session
    starts with a clean demo list.  Restoring old demos causes "mixing"
    because the transcript store may contain demos from unrelated previous
    sessions.  Re-record demos in each new session.
    """
    window_us = window_min * 60 * 1_000_000
    log.info("restoring observations from transcript-mcp  window=%d min", window_min)

    obs_entries = await transcript.query_recent(source + ":observations", window_us)
    restored = 0
    for entry in obs_entries:
        ts  = entry.get("timestamp_us", 0)
        txt = entry.get("text", "").strip()
        if txt:
            memory.restore_observation(Observation(
                timestamp_us = ts,
                description  = txt,
                image_path   = "",
            ))
            restored += 1
    log.info("restored %d observations", restored)


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
        "instructions":  demo.instructions,
        "voice_notes": [
            {"timestamp_us": v.timestamp_us, "text": v.text}
            for v in demo.voice_notes
        ],
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
    _trace_path = str(pathlib.Path(_run_dir) / "trace.log") if _run_dir else "/tmp/glasses-agent-nat-trace.log"
    _setup_trace_log(_trace_path)

    log.info("glasses-agent-nat starting")

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

    nat_runtime = await NatRuntime.create(cfg.nat_workflow_config)
    agent: GlassesAgent | None = None
    query_proc: QueryProcessor | None = None
    transcript: TranscriptClient | None = None
    try:
        if ready_file:
            ready_file.touch()

        # ── memory + transcript ───────────────────────────────────────────────
        memory     = AgentMemory(max_obs=cfg.vlm_obs_max)
        transcript = TranscriptClient(nat_runtime)

        # Restore from transcript-mcp.
        await _restore_memory(memory, transcript, cfg.transcript_source)

        # ── query processor ───────────────────────────────────────────────────

        async def _say(pid: str, text: str) -> None:
            if agent is None:
                return
            await agent.say(pid, text)

        async def _send_text(pid: str, text: str, topic: str) -> None:
            if agent is None:
                return
            await agent.send_text(pid, text, topic)

        async def _flush_audio(pid: str) -> None:
            if agent is None:
                return
            await agent.flush_audio(pid)

        query_proc = QueryProcessor(
            cfg           = cfg,
            memory        = memory,
            nat_runtime   = nat_runtime,
            send_text     = _send_text,
            say           = _say,
            flush_audio   = _flush_audio,
        )

        # ── main agent ────────────────────────────────────────────────────────
        agent = GlassesAgent(
            cfg               = cfg,
            memory            = memory,
            transcript_client = transcript,
            query_processor   = query_proc,
            stt_url           = cfg.stt_server,
            tts_url           = cfg.tts_server,
            nat_runtime       = nat_runtime,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, agent.shutdown)

        log.info("glasses-agent-nat running")
        try:
            await agent.run()
        finally:
            if agent is not None:
                agent.shutdown()
    finally:
        if query_proc is not None:
            await query_proc.close()
        if transcript is not None:
            await transcript.close()
        await nat_runtime.close()

    log.info("glasses-agent-nat stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    setup_logging("worker", namespace="glasses-agent-nat")
    cfg = load_config(ns.config)
    asyncio.run(main(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
