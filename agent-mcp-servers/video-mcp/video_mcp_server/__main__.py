"""
Video MCP server.

Reads the video recording directory written by the XR-Media-Hub recorder
directly from disk — no HTTP dependency on the hub at runtime.

On-disk layout (written by the hub's VideoRecorder):
    <recordings_dir>/<participant_id>/
        <start_us>.264   — raw H.264 Annex B, starts with IDR
        <start_us>.json  — sidecar: start_us, end_us, num_frames, width, height, size_bytes

MCP tools (FastMCP, StreamableHTTP on /mcp)
───────────────────────────────────────────
  get_video_stats(participant_id)
      Summary: num_chunks, total_bytes, avg_chunk_bytes, earliest_us, latest_us.

  query_video(participant_id, start_us, end_us)
      Concatenate all chunks that overlap [start_us, end_us], write to a file,
      return the path and metadata.

  list_recording_participants()
      Participant IDs that have at least one chunk on disk.

HTTP endpoints
──────────────
  GET /stats/{participant_id}   — same data as get_video_stats, for workers
  GET /health

Config (video_mcp_server.yaml)
───────────────────────────────
    recordings_dir: /tmp/xr_recordings   # must match hub video_recording.out_dir
    out_dir:        /tmp/xr_video_queries
    host:           0.0.0.0
    port:           8210
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastmcp import FastMCP

log = logging.getLogger("video_mcp_server")


# ── chunk store (reads the recording directory) ───────────────────────────────

class ChunkStore:
    def __init__(self, recordings_dir: pathlib.Path) -> None:
        self._root = recordings_dir

    def _pid_dir(self, pid: str) -> pathlib.Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)
        return self._root / safe

    def list_participants(self) -> list[str]:
        if not self._root.exists():
            return []
        return [d.name for d in sorted(self._root.iterdir()) if d.is_dir()]

    def _sorted_chunks(self, pid: str) -> list[pathlib.Path]:
        pid_dir = self._pid_dir(pid)
        if not pid_dir.exists():
            return []
        return sorted(pid_dir.glob("*.264"), key=lambda p: int(p.stem))

    def _load_meta(self, h264: pathlib.Path) -> dict:
        meta_path = h264.with_suffix(".json")
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        # Fall back to filename stem if sidecar is missing.
        return {"start_us": int(h264.stem), "end_us": int(h264.stem), "size_bytes": h264.stat().st_size}

    def stats(self, pid: str) -> dict | None:
        chunks = self._sorted_chunks(pid)
        if not chunks:
            return None
        metas      = [self._load_meta(c) for c in chunks]
        total      = sum(m.get("size_bytes", c.stat().st_size) for m, c in zip(metas, chunks))
        return {
            "participant_id":  pid,
            "num_chunks":      len(chunks),
            "total_bytes":     total,
            "avg_chunk_bytes": total // len(chunks),
            "earliest_us":     metas[0].get("start_us",  int(chunks[0].stem)),
            "latest_us":       metas[-1].get("end_us",   int(chunks[-1].stem)),
        }

    def query(self, pid: str, start_us: int, end_us: int) -> bytes | None:
        chunks = self._sorted_chunks(pid)
        if not chunks:
            return None

        metas = [(c, self._load_meta(c)) for c in chunks]

        # Include the last chunk that started before the window (gives us an IDR
        # at the start of the result) plus all chunks that overlap the window.
        anchor  = None
        overlap = []
        for h264, meta in metas:
            cs = meta.get("start_us", int(h264.stem))
            ce = meta.get("end_us",   cs)
            if cs <= end_us and ce >= start_us:
                overlap.append(h264)
            elif cs < start_us:
                anchor = h264   # latest chunk entirely before the window

        selected = ([anchor] if anchor and not overlap else []) + overlap
        if not selected:
            return None

        return b"".join(p.read_bytes() for p in selected)


# ── app ───────────────────────────────────────────────────────────────────────

def build_app(store: ChunkStore, out_dir: pathlib.Path) -> FastAPI:
    app = FastAPI(title="Video MCP Server", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "recordings_dir": str(store._root)}

    @app.get("/stats/{participant_id}")
    async def http_stats(participant_id: str) -> JSONResponse:
        result = store.stats(participant_id)
        if result is None:
            raise HTTPException(404, f"No video chunks for {participant_id!r}")
        return JSONResponse(result)

    mcp = build_mcp(store, out_dir)
    app.mount("/mcp", mcp.http_app())
    return app


def build_mcp(store: ChunkStore, out_dir: pathlib.Path) -> "FastMCP":
    """Return a composed FastMCP server with all video tools bound to *store*."""
    mcp = FastMCP("video-mcp")

    @mcp.tool()
    def list_recording_participants() -> list[str]:
        """Return participant IDs that have recorded video chunks on disk."""
        return store.list_participants()

    @mcp.tool()
    def get_video_stats(participant_id: str) -> dict:
        """
        Summary statistics for all recorded chunks of *participant_id*.

        Keys: participant_id, num_chunks, total_bytes, avg_chunk_bytes,
              earliest_us (Unix µs), latest_us (Unix µs).
        Returns an error dict if no chunks exist.
        """
        result = store.stats(participant_id)
        if result is None:
            return {"error": f"No video chunks for {participant_id!r}"}
        return result

    @mcp.tool()
    def query_video(participant_id: str, start_us: int, end_us: int) -> dict:
        """
        Concatenate H.264 chunks for *participant_id* covering [start_us, end_us]
        (Unix microseconds), write to a file, and return the path.

        The result is a raw H.264 Annex B stream starting with an IDR frame.
        Keys: path (str), size (int), start_us (int), end_us (int).
        """
        data = store.query(participant_id, start_us, end_us)
        if data is None:
            return {"error": f"No video chunks for {participant_id!r} in requested window"}
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in participant_id)
        out_path = out_dir / f"{safe}_{start_us}_{end_us}.264"
        out_path.write_bytes(data)
        log.info("query_video  pid=%r  %d–%d  %d bytes → %s",
                 participant_id, start_us, end_us, len(data), out_path)
        return {"path": str(out_path), "size": len(data), "start_us": start_us, "end_us": end_us}

    return mcp


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

    recordings_dir = pathlib.Path(cfg.get("recordings_dir", "/tmp/xr_recordings"))
    out_dir        = pathlib.Path(cfg.get("out_dir",        "/tmp/xr_video_queries"))
    host           = cfg.get("host", "0.0.0.0")
    port           = int(cfg.get("port", 8210))

    out_dir.mkdir(parents=True, exist_ok=True)

    store = ChunkStore(recordings_dir)
    app   = build_app(store, out_dir)

    log.info("video-mcp-server  recordings_dir=%s  port=%d", recordings_dir, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
