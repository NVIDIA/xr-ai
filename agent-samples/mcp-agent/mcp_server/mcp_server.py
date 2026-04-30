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
import pathlib

import uvicorn
import yaml

from app import build, build_app  # noqa: F401  (re-exported for tests)

log = logging.getLogger("mcp_server")


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

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(_serve(cfg))


if __name__ == "__main__":
    run()
