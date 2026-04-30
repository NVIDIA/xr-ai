"""
Composed MCP server for the mcp-agent example.

Pure FastMCP — mounts two sub-servers (transcript, video) into a single
FastMCP instance and serves the StreamableHTTP transport at /mcp. There
are no REST endpoints; workers use ``fastmcp.Client``.

Config (mcp_server.yaml)
-------------------------
    host: 0.0.0.0
    port: 8200

    transcript:
      transcripts_dir: /tmp/xr_transcripts/mcp-agent

    video:
      recordings_dir:  /tmp/xr_recordings/mcp-agent   # must match hub out_dir
      out_dir:         /tmp/xr_video_queries/mcp-agent
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import uvicorn
import yaml
from fastmcp import FastMCP

from transcript_mcp_server import TranscriptStore, build_mcp as build_transcript_mcp
from video_mcp_server      import ChunkStore,      build_mcp as build_video_mcp

log = logging.getLogger("mcp_server")


def build_app(cfg: dict):
    """Compose transcript + video MCP servers into one FastMCP and return its
    ASGI app, served at /mcp."""
    transcripts_dir = pathlib.Path(cfg.get("transcript", {}).get("transcripts_dir", "/tmp/xr_transcripts"))
    recordings_dir  = pathlib.Path(cfg.get("video",      {}).get("recordings_dir",  "/tmp/xr_recordings"))
    out_dir         = pathlib.Path(cfg.get("video",      {}).get("out_dir",         "/tmp/xr_video_queries"))
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = TranscriptStore(str(transcripts_dir))
    chunks      = ChunkStore(recordings_dir)

    mcp = FastMCP("xr-mcp")
    mcp.mount(build_transcript_mcp(transcripts),  namespace="transcript")
    mcp.mount(build_video_mcp(chunks, out_dir),   namespace="video")

    return mcp.http_app(path="/mcp")


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

    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8200))

    log.info("xr-mcp-server  port=%d", port)
    uvicorn.run(build_app(cfg), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
