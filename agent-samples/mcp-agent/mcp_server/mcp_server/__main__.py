"""
Composed MCP server for the mcp-agent example.

Mounts three sub-servers (transcript, video, client-control) into a single
FastMCP instance. All access — by workers and by external LLMs alike — is via
MCP tools at /mcp. The only REST endpoint is /health for startup probing.

Config (mcp_server.yaml)
-------------------------
    host: 0.0.0.0
    port: 8200

    transcript:
      transcripts_dir: /tmp/xr_transcripts/mcp-agent

    video:
      recordings_dir:  /tmp/xr_recordings/mcp-agent   # must match hub out_dir
      out_dir:         /tmp/xr_video_queries/mcp-agent

    client_control:
      hub_pub:  ipc:///tmp/xr_hub_pub
      hub_push: ipc:///tmp/xr_hub_in
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import uvicorn
import yaml
from fastapi import FastAPI
from fastmcp import FastMCP

from client_control_mcp_server import ClientControlBridge, build_mcp as build_client_control_mcp
from transcript_mcp_server     import TranscriptStore,      build_mcp as build_transcript_mcp
from video_mcp_server          import ChunkStore,           build_mcp as build_video_mcp

log = logging.getLogger("mcp_server")


def build_app(cfg: dict) -> FastAPI:
    transcripts_dir = pathlib.Path(cfg.get("transcript", {}).get("transcripts_dir", "/tmp/xr_transcripts"))
    recordings_dir  = pathlib.Path(cfg.get("video",      {}).get("recordings_dir",  "/tmp/xr_recordings"))
    out_dir         = pathlib.Path(cfg.get("video",      {}).get("out_dir",         "/tmp/xr_video_queries"))
    cc_cfg          = cfg.get("client_control", {})
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = TranscriptStore(str(transcripts_dir))
    chunks      = ChunkStore(recordings_dir)
    cc_bridge   = ClientControlBridge(
        hub_pub  = cc_cfg.get("hub_pub",  "ipc:///tmp/xr_hub_pub"),
        hub_push = cc_cfg.get("hub_push", "ipc:///tmp/xr_hub_in"),
    )

    mcp = FastMCP("xr-mcp")
    mcp.mount(build_transcript_mcp(transcripts),       namespace="transcript")
    mcp.mount(build_video_mcp(chunks, out_dir),        namespace="video")
    mcp.mount(build_client_control_mcp(cc_bridge),     namespace="client_control")

    mcp_app = mcp.http_app(path="/")
    app = FastAPI(title="XR MCP Server", docs_url=None, redoc_url=None,
                  lifespan=mcp_app.lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    app.mount("/mcp", mcp_app)
    return app


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
