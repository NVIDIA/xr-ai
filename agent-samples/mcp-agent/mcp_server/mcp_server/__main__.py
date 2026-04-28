"""
Composed MCP server for the mcp-agent example.

Demonstrates FastMCP composition: reads a ``skills`` block from YAML and
mounts the requested sub-servers into a single FastMCP instance.

Available skills
----------------
  transcript — stores and queries timestamped speech transcripts
  video      — queries NVENC-recorded H.264 video chunks from disk

Config (mcp_server.yaml)
-------------------------
    host:  0.0.0.0
    port:  8200

    skills:
      transcript:
        transcripts_dir: /tmp/xr_transcripts/mcp-agent
      video:
        recordings_dir:  /tmp/xr_recordings/mcp-agent   # must match hub out_dir
        out_dir:         /tmp/xr_video_queries/mcp-agent

HTTP endpoints
--------------
  POST /ingest                       — worker pushes transcripts (requires transcript skill)
  GET  /transcript/stats/{pid}       — transcript stats for worker (requires transcript skill)
  GET  /video/stats/{pid}            — video stats for worker     (requires video skill)
  GET  /health

MCP endpoint (StreamableHTTP): /mcp
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel

from transcript_mcp_server import TranscriptStore, build_mcp as build_transcript_mcp
from video_mcp_server import ChunkStore, build_mcp as build_video_mcp

log = logging.getLogger("mcp_server")


class IngestRequest(BaseModel):
    participant_id: str
    timestamp_us:   int
    text:           str


def build_app(cfg: dict) -> FastAPI:
    skills = cfg.get("skills", {})

    # ── skill: transcript ─────────────────────────────────────────────────────

    transcript_store: TranscriptStore | None = None
    if "transcript" in skills:
        skill_cfg = skills["transcript"]
        transcript_store = TranscriptStore(
            skill_cfg.get("transcripts_dir", "/tmp/xr_transcripts")
        )
        log.info("skill transcript  dir=%s", skill_cfg.get("transcripts_dir"))

    # ── skill: video ──────────────────────────────────────────────────────────

    chunk_store: ChunkStore | None = None
    if "video" in skills:
        skill_cfg = skills["video"]
        recordings_dir = pathlib.Path(skill_cfg.get("recordings_dir", "/tmp/xr_recordings"))
        out_dir        = pathlib.Path(skill_cfg.get("out_dir",        "/tmp/xr_video_queries"))
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_store = ChunkStore(recordings_dir)
        log.info("skill video  recordings=%s", recordings_dir)

    # ── FastMCP composition ───────────────────────────────────────────────────

    mcp = FastMCP("xr-mcp")
    if transcript_store is not None:
        mcp.mount(build_transcript_mcp(transcript_store), namespace="transcript")
    if chunk_store is not None:
        mcp.mount(build_video_mcp(chunk_store, out_dir), namespace="video")

    # ── FastAPI wrapper ───────────────────────────────────────────────────────

    app = FastAPI(title="XR MCP Server", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "skills": list(skills)}

    if transcript_store is not None:
        @app.post("/ingest")
        async def ingest(req: IngestRequest) -> JSONResponse:
            if not req.text.strip():
                raise HTTPException(400, "text must not be empty")
            transcript_store.append(req.participant_id, req.timestamp_us, req.text)
            log.info("ingest  pid=%r  ts=%d  %r",
                     req.participant_id, req.timestamp_us, req.text[:80])
            return JSONResponse({"ok": True})

        @app.get("/transcript/stats/{participant_id}")
        async def transcript_stats(participant_id: str) -> JSONResponse:
            result = transcript_store.stats(participant_id)
            if result is None:
                raise HTTPException(404, f"No transcripts for {participant_id!r}")
            return JSONResponse(result)

    if chunk_store is not None:
        @app.get("/video/stats/{participant_id}")
        async def video_stats(participant_id: str) -> JSONResponse:
            result = chunk_store.stats(participant_id)
            if result is None:
                raise HTTPException(404, f"No video chunks for {participant_id!r}")
            return JSONResponse(result)

    app.mount("/mcp", mcp.http_app())
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

    app = build_app(cfg)
    log.info("xr-mcp-server  skills=%s  port=%d", list(cfg.get("skills", {})), port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
