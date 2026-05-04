# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MCP agent worker — entry point.

Launched as a subprocess by ``uv run mcp_agent`` (the orchestrator).
Do not run this directly.

What it does
------------
1. Listens for audio from XR clients via the hub IPC.
2. Runs VAD to detect speech boundaries (same logic as echo-agent).
3. At end of each utterance, runs STT on the full audio buffer.
4. Calls the ``transcript_add_transcript`` MCP tool with
   ``source_id=<participant_id>`` to record the utterance.  (The
   transcript store is keyed by an arbitrary ``source_id`` string —
   live participant identities here, but agents can also write under
   internal source names like ``"agent-vlm"``.)
5. On any data-channel message, calls ``transcript_get_transcript_stats``
   and ``video_get_video_stats`` and sends a summary back on topic
   ``mcp.stats``.

The composed mcp-server is a pure-FastMCP process at /mcp (no REST). It
mounts the transcript and video sub-servers under their respective
namespaces. External LLM agents can connect to the same /mcp to call
tools directly.

Config (mcp_agent_worker.yaml — auto-passed by the launcher)
-------------------------------------------------------------
    stt_server:        http://localhost:8103
    mcp_server:        http://localhost:8200
    silence_threshold: 0.01
    silence_duration:  0.8
    min_speech:        0.3
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import signal

import yaml

from xr_ai_agent import ProcessorEndpoint

from agent import McpAgent
from services import SttClient, wait_for_services

log = logging.getLogger("mcp_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


def _resolve_log_level(cfg: dict) -> str:
    """Per-process YAML log_level > XR_AI_LOG_LEVEL env > INFO. Inlined to
    keep workers stdlib-only and to avoid importing from xr_ai_launcher
    (forbidden for workers per AGENTS.md)."""
    val = cfg.get("log_level")
    if val and isinstance(val, str):
        v = val.upper()
        if v in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
            return v
    env = os.environ.get("XR_AI_LOG_LEVEL", "").upper()
    if env in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
        return env
    return "INFO"


def _resolve_http_log_level(cfg: dict) -> str:
    """Per-process YAML http_log_level > WARNING. Controls httpx + httpcore
    loggers (per-HTTP-request noise). Independent of the main `log_level`
    field; file capture is unaffected (DEBUG always)."""
    val = cfg.get("http_log_level")
    if val and isinstance(val, str):
        v = val.upper()
        if v in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
            return v
    return "WARNING"


# ── per-source level filters (file gets DEBUG always; terminal at user level) ─

class _AboveThresholdFilter(logging.Filter):
    """Pass records at level >= per-source threshold (terminal display)."""
    def __init__(self, default: int, sources: dict | None = None) -> None:
        super().__init__()
        self.default = default
        self.sources = sources or {}
    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, thr in self.sources.items():
            if record.name.startswith(prefix):
                return record.levelno >= thr
        return record.levelno >= self.default


class _BelowThresholdFilter(logging.Filter):
    """Pass records at level < per-source threshold (file capture for the
    levels the StreamHandler dropped)."""
    def __init__(self, default: int, sources: dict | None = None) -> None:
        super().__init__()
        self.default = default
        self.sources = sources or {}
    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, thr in self.sources.items():
            if record.name.startswith(prefix):
                return record.levelno < thr
        return record.levelno < self.default


def _setup_logging(cfg: dict, sources: dict | None = None) -> None:
    """Multi-handler setup: terminal at user level (per-source-aware via
    AboveThresholdFilter), file at DEBUG via FileHandlers with the inverse
    BelowThresholdFilter — exclusive routing, no duplicates with the
    launcher's PIPE tee.  Reads XR_AI_LOG_DIR + XR_AI_LOG_NAME env vars
    (set by the launcher) to pick file paths; degrades to terminal-only
    when unset."""
    user_level = getattr(logging, _resolve_log_level(cfg), logging.INFO)
    sources = sources or {}
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    sh.addFilter(_AboveThresholdFilter(user_level, sources))
    root.addHandler(sh)

    log_dir  = os.environ.get("XR_AI_LOG_DIR")
    log_name = os.environ.get("XR_AI_LOG_NAME")
    if log_dir and log_name:
        for path in (f"{log_dir}/{log_name}.log", f"{log_dir}/combined.log"):
            try:
                fh = logging.FileHandler(path, mode="a")
            except OSError:
                continue
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            fh.addFilter(_BelowThresholdFilter(user_level, sources))
            root.addHandler(fh)


async def main(cfg: dict) -> None:
    http_level = getattr(logging, _resolve_http_log_level(cfg), logging.WARNING)
    _setup_logging(cfg, sources={"httpx": http_level, "httpcore": http_level})

    stt     = SttClient(cfg.get("stt_server", "http://localhost:8103"))
    mcp_url = cfg.get("mcp_server", "http://localhost:8200").rstrip("/") + "/mcp"
    await wait_for_services(stt.health_url, mcp_url)

    ep    = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    agent = McpAgent(
        ep, stt, mcp_url,
        silence_threshold=float(cfg.get("silence_threshold", 0.01)),
        silence_duration =float(cfg.get("silence_duration",  0.8)),
        min_speech       =float(cfg.get("min_speech",        0.3)),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("mcp-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    log.info("mcp-agent stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
