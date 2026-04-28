"""
Transcript MCP server.

Two interfaces on a single port:

  HTTP ingest (non-MCP)
  ─────────────────────
  POST /ingest
      Body: {"participant_id": "...", "timestamp_us": 1234, "text": "..."}
      Returns 200 {"ok": true}.
      Called by the mcp-agent worker after each STT transcription.

  MCP (FastMCP, mounted at /mcp)
  ──────────────────────────────
  Tool: query_transcripts(participant_id, start_us, end_us) → list[Transcript]
      Returns all stored transcript segments for *participant_id* whose
      timestamp falls within [start_us, end_us] (Unix microseconds).

  Tool: list_participants() → list[str]
      Returns all participant IDs that have at least one transcript stored.

Storage
───────
Transcripts are stored per-participant in JSONL files:

    <transcripts_dir>/<participant_id>.jsonl

Each line is a JSON object {"timestamp_us": int, "text": str}.
Files persist across server restarts.

Config (transcript_mcp_server.yaml)
────────────────────────────────────
    transcripts_dir: /tmp/xr_transcripts
    host:            0.0.0.0
    port:            8200
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel

log = logging.getLogger("transcript_mcp_server")


# ── data models ───────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    participant_id: str
    timestamp_us:   int
    text:           str


class Transcript(BaseModel):
    participant_id: str
    timestamp_us:   int
    text:           str


# ── storage ───────────────────────────────────────────────────────────────────

class TranscriptStore:
    """Append-only JSONL storage for timestamped transcript segments."""

    def __init__(self, transcripts_dir: str) -> None:
        self._dir = pathlib.Path(transcripts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, pid: str) -> pathlib.Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)
        return self._dir / f"{safe}.jsonl"

    def append(self, pid: str, timestamp_us: int, text: str) -> None:
        record = json.dumps({"timestamp_us": timestamp_us, "text": text})
        with self._path(pid).open("a") as f:
            f.write(record + "\n")

    def query(self, pid: str, start_us: int, end_us: int) -> list[dict]:
        path = self._path(pid)
        if not path.exists():
            return []
        results = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if start_us <= rec["timestamp_us"] <= end_us:
                        results.append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
        return results

    def list_participants(self) -> list[str]:
        return [p.stem for p in sorted(self._dir.glob("*.jsonl"))]

    def stats(self, pid: str) -> dict | None:
        path = self._path(pid)
        if not path.exists():
            return None
        count = earliest = latest = total_chars = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts  = rec["timestamp_us"]
                    count += 1
                    total_chars += len(rec.get("text", ""))
                    if earliest == 0 or ts < earliest:
                        earliest = ts
                    if ts > latest:
                        latest = ts
                except (json.JSONDecodeError, KeyError):
                    continue
        return {
            "participant_id": pid,
            "count":          count,
            "total_chars":    total_chars,
            "earliest_us":    earliest,
            "latest_us":      latest,
        }


# ── server ────────────────────────────────────────────────────────────────────

def build_app(store: TranscriptStore) -> FastAPI:
    app = FastAPI(title="Transcript MCP Server", docs_url=None, redoc_url=None)

    # ── HTTP ingest endpoint ──────────────────────────────────────────────────

    @app.post("/ingest")
    async def ingest(req: IngestRequest) -> JSONResponse:
        if not req.text.strip():
            raise HTTPException(400, "text must not be empty")
        store.append(req.participant_id, req.timestamp_us, req.text)
        log.info("ingest  pid=%r  ts=%d  %r", req.participant_id, req.timestamp_us, req.text[:80])
        return JSONResponse({"ok": True})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/stats/{participant_id}")
    async def stats(participant_id: str) -> JSONResponse:
        result = store.stats(participant_id)
        if result is None:
            raise HTTPException(404, f"No transcripts for {participant_id!r}")
        return JSONResponse(result)

    mcp = build_mcp(store)
    app.mount("/mcp", mcp.http_app())

    return app


def build_mcp(store: TranscriptStore) -> "FastMCP":
    """Return a composed FastMCP server with all transcript tools bound to *store*."""
    mcp = FastMCP("transcript-mcp")

    @mcp.tool()
    def query_transcripts(
        participant_id: str,
        start_us: int,
        end_us: int,
    ) -> list[dict]:
        """
        Return transcript segments for *participant_id* in the time window
        [start_us, end_us] (Unix microseconds).

        Each result has keys: timestamp_us (int), text (str).
        Results are ordered by timestamp_us ascending.
        """
        results = store.query(participant_id, start_us, end_us)
        results.sort(key=lambda r: r["timestamp_us"])
        return results

    @mcp.tool()
    def list_participants() -> list[str]:
        """Return all participant IDs that have stored transcripts."""
        return store.list_participants()

    @mcp.tool()
    def get_transcript_stats(participant_id: str) -> dict:
        """
        Return summary statistics for a participant's stored transcripts.

        Keys: participant_id, count (utterances), total_chars,
              earliest_us (Unix µs), latest_us (Unix µs).
        Returns an error dict if no transcripts exist.
        """
        result = store.stats(participant_id)
        if result is None:
            return {"error": f"No transcripts for {participant_id!r}"}
        return result

    return mcp


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    transcripts_dir = cfg.get("transcripts_dir", "/tmp/xr_transcripts")
    host            = cfg.get("host", "0.0.0.0")
    port            = int(cfg.get("port", 8200))

    store = TranscriptStore(transcripts_dir)
    app   = build_app(store)

    log.info(
        "transcript-mcp-server  ingest=POST /ingest  mcp=/mcp  port=%d  dir=%s",
        port, transcripts_dir,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
