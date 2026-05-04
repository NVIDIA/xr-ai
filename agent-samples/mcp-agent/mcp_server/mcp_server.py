# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Composed MCP server for the mcp-agent example.

Pure FastMCP — mounts two sub-servers (transcript, video) into a single
FastMCP instance and serves the StreamableHTTP transport at /mcp. There
are no REST endpoints; workers use ``fastmcp.Client``.

Config (mcp_server.yaml — auto-passed by the launcher)
-------------------------------------------------------
    host: 0.0.0.0
    port: 8200

    transcript:
      transcripts_dir: /tmp/xr_transcripts/mcp-agent

    video:
      recordings_dir:  /dev/shm/xr-ai/recordings   # must match hub out_dir
      out_dir:         /tmp/xr_video_queries/mcp-agent
      hub_pub:         ipc:///tmp/xr_hub_pub
      hub_push:        ipc:///tmp/xr_hub_in
      gpu_id:          0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib

import uvicorn
import yaml

from app import build, build_app  # noqa: F401  (re-exported for tests)

log = logging.getLogger("mcp_server")


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


async def _serve(cfg: dict) -> None:
    app, ep = build(cfg)

    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8200))
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    ep_task = asyncio.create_task(ep.run(), name="composed_mcp_processor")
    log.info("xr-mcp-server  port=%d", port)
    try:
        await server.serve()
    finally:
        ep.stop()
        ep_task.cancel()
        try:
            await ep_task
        except (asyncio.CancelledError, Exception):
            pass
        ep.close()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    logging.basicConfig(
        level=getattr(logging, _resolve_log_level(cfg), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    asyncio.run(_serve(cfg))


if __name__ == "__main__":
    run()
