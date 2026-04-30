"""
Composed MCP server for the mcp-agent example.

Pure FastMCP — mounts two sub-servers (transcript, video) into a single
FastMCP instance and serves the StreamableHTTP transport at /mcp. There
are no REST endpoints; workers use ``fastmcp.Client``.

The video sub-server connects to the hub as a ``ProcessorEndpoint`` to
serve live latest-frame queries; we manage the endpoint task in the same
asyncio loop as the uvicorn server.

Config (mcp_server.yaml)
-------------------------
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
from fastmcp import FastMCP

from transcript_mcp_server import TranscriptStore, build_mcp as build_transcript_mcp
from video_mcp_server      import (ChunkStore, FrameProvider,
                                   build_mcp as build_video_mcp)
from xr_ai_agent           import ProcessorEndpoint, Subscribe

log = logging.getLogger("mcp_server")


def _build(cfg: dict) -> tuple[object, ProcessorEndpoint]:
    """Build the composed FastMCP ASGI app and return it alongside the
    ProcessorEndpoint that backs live frame queries (caller manages its
    lifecycle)."""
    transcript_cfg = cfg.get("transcript", {})
    video_cfg      = cfg.get("video",      {})

    transcripts_dir = pathlib.Path(transcript_cfg.get("transcripts_dir", "/tmp/xr_transcripts"))
    recordings_dir  = pathlib.Path(video_cfg.get("recordings_dir",  "/dev/shm/xr-ai/recordings"))
    out_dir         = pathlib.Path(video_cfg.get("out_dir",         "/tmp/xr_video_queries"))
    hub_pub         = video_cfg.get("hub_pub",  "ipc:///tmp/xr_hub_pub")
    hub_push        = video_cfg.get("hub_push", "ipc:///tmp/xr_hub_in")
    gpu_id          = int(video_cfg.get("gpu_id", 0))
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = TranscriptStore(str(transcripts_dir))
    chunks      = ChunkStore(recordings_dir)

    ep       = ProcessorEndpoint(
        sub_addr=hub_pub, push_addr=hub_push,
        filter=Subscribe.VIDEO,
    )
    provider = FrameProvider(ep)

    mcp = FastMCP("xr-mcp")
    mcp.mount(build_transcript_mcp(transcripts),                          namespace="transcript")
    mcp.mount(build_video_mcp(chunks, out_dir, provider, gpu_id=gpu_id),  namespace="video")

    return mcp.http_app(path="/mcp"), ep


def build_app(cfg: dict):
    """Backwards-compatible entry: returns just the ASGI app (no
    endpoint lifecycle). Used by tests; production should call
    ``_build`` and manage the endpoint."""
    app, _ep = _build(cfg)
    return app


async def _serve(cfg: dict) -> None:
    app, ep = _build(cfg)

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
