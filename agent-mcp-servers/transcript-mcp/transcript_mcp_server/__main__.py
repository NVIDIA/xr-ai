# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transcript MCP server.

Pure FastMCP — every operation is an MCP tool at /mcp. There are no REST
endpoints. Workers use ``fastmcp.Client`` (or any MCP client) to ingest
transcripts and query stats.

Source IDs
──────────
Every record is keyed by a free-form ``source_id`` string. Sources can be
real LiveKit participant identities (e.g. ``"alice@home"``,
``"ipad-pro-1"``), or internal/synthetic names (e.g. ``"agent-vlm"``,
``"tts"``) — the store doesn't interpret the value, it just keys storage
by it. Filesystem paths are sanitized internally; the original ``source_id``
is recovered from a ``.identity`` sidecar so list/query round-trip cleanly.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  query_transcripts(source_id, start_us, end_us) → list[dict]
      Return all stored transcript segments for *source_id* whose
      timestamp falls within [start_us, end_us] (Unix microseconds).

  add_transcript(source_id, timestamp_us, text) → dict
      Append a transcript segment for *source_id*. Returns
      ``{"ok": true}`` or ``{"error": ...}`` if *text* is empty.

  list_sources() → list[str]
      All source IDs that have at least one stored transcript.

  get_transcript_stats(source_id) → dict
      Summary statistics (count, total_chars, earliest_us, latest_us).

Storage
───────
Per-source JSONL alongside a ``.identity`` sidecar holding the raw name:

    <transcripts_dir>/<safe>.jsonl
    <transcripts_dir>/<safe>.identity

Each JSONL line is ``{"timestamp_us": int, "text": str}``. Files persist
across server restarts. Distinct ``source_id`` values that map to the
same ``_safe_name`` get a counter suffix (``alice_home``, ``alice_home_2``…)
so they don't share storage.

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

import uvicorn
import yaml
from fastmcp import FastMCP

log = logging.getLogger("transcript_mcp_server")


def _safe_name(s: str) -> str:
    """Filesystem-safe version of *s*."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# ── storage ───────────────────────────────────────────────────────────────────

class TranscriptStore:
    """Append-only JSONL storage for timestamped transcript segments.

    Records are keyed by ``source_id`` — any string the caller chooses.
    Sanitization for filesystem paths is internal; the raw ``source_id``
    is preserved in a ``.identity`` sidecar.
    """

    def __init__(self, transcripts_dir: str) -> None:
        self._dir = pathlib.Path(transcripts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Resolve once at construction so the safe root can't be swapped
        # for a symlink (TOCTOU) between subsequent _check calls.
        self._root = self._dir.resolve()

    # ── path resolution ──────────────────────────────────────────────

    def _check(self, path: pathlib.Path) -> pathlib.Path:
        if not path.resolve().is_relative_to(self._root):
            raise ValueError(f"Path escapes transcript directory: {path}")
        return path

    def _resolve_or_create(self, source_id: str) -> pathlib.Path:
        """Resolve *source_id* to ``<dir>/<stem>.jsonl``, creating the
        ``.identity`` sidecar on first use. Disambiguates collisions
        with a counter suffix."""
        safe   = _safe_name(source_id)
        suffix = 1
        while True:
            stem  = safe if suffix == 1 else f"{safe}_{suffix}"
            ident = self._dir / f"{stem}.identity"
            jsonl = self._dir / f"{stem}.jsonl"
            if not ident.exists() and not jsonl.exists():
                ident.write_text(source_id, encoding="utf-8")
                return self._check(jsonl)
            if ident.exists() and ident.read_text(encoding="utf-8") == source_id:
                return self._check(jsonl)
            # Legacy pre-sidecar: jsonl exists, no sidecar, raw == safe.
            if (
                not ident.exists() and jsonl.exists()
                and source_id == safe and suffix == 1
            ):
                ident.write_text(source_id, encoding="utf-8")
                return self._check(jsonl)
            suffix += 1

    def _resolve_existing(self, source_id: str) -> pathlib.Path | None:
        """Return the JSONL path for *source_id* if any record has been
        written for it, else ``None`` — never creates anything."""
        if not self._dir.exists():
            return None
        safe = _safe_name(source_id)
        # Fast path: canonical name.
        canonical_jsonl = self._dir / f"{safe}.jsonl"
        canonical_ident = self._dir / f"{safe}.identity"
        if canonical_jsonl.exists():
            if canonical_ident.exists():
                if canonical_ident.read_text(encoding="utf-8").strip() == source_id.strip():
                    jsonl = self._check(canonical_jsonl)
                    return jsonl
            elif source_id == safe:
                jsonl = self._check(canonical_jsonl)
                return jsonl
        # Slow path: scan sidecars.
        for ident in sorted(self._dir.glob("*.identity")):
            if ident.read_text(encoding="utf-8").strip() == source_id.strip():
                jsonl = self._check(ident.with_suffix(".jsonl"))
                return jsonl if jsonl.exists() else None
        return None

    # ── operations ───────────────────────────────────────────────────

    def append(self, source_id: str, timestamp_us: int, text: str) -> None:
        record = json.dumps({"timestamp_us": timestamp_us, "text": text})
        with self._resolve_or_create(source_id).open("a") as f:
            f.write(record + "\n")

    def query(self, source_id: str, start_us: int, end_us: int) -> list[dict]:
        path = self._resolve_existing(source_id)
        if path is None:
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

    def list_sources(self) -> list[str]:
        """Return raw source IDs (read from ``.identity`` sidecars; falls
        back to file stem for legacy pre-sidecar files)."""
        if not self._dir.exists():
            return []
        out: list[str] = []
        seen_stems: set[str] = set()
        for ident in sorted(self._dir.glob("*.identity")):
            out.append(ident.read_text(encoding="utf-8"))
            seen_stems.add(ident.stem)
        for jsonl in sorted(self._dir.glob("*.jsonl")):
            if jsonl.stem not in seen_stems:
                out.append(jsonl.stem)
        return out

    def stats(self, source_id: str) -> dict | None:
        path = self._resolve_existing(source_id)
        if path is None:
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
            "source_id":   source_id,
            "count":       count,
            "total_chars": total_chars,
            "earliest_us": earliest,
            "latest_us":   latest,
        }


# ── server ────────────────────────────────────────────────────────────────────

def build_mcp(store: TranscriptStore) -> "FastMCP":
    """Return a composed FastMCP server with all transcript tools bound to *store*."""
    mcp = FastMCP("transcript-mcp")

    @mcp.tool()
    def query_transcripts(
        source_id: str,
        start_us:  int,
        end_us:    int,
    ) -> list[dict]:
        """
        Return transcript segments for *source_id* in the time window
        [start_us, end_us] (Unix microseconds).

        Each result has keys: timestamp_us (int), text (str).
        Results are ordered by timestamp_us ascending.
        """
        results = store.query(source_id, start_us, end_us)
        results.sort(key=lambda r: r["timestamp_us"])
        return results

    @mcp.tool()
    def add_transcript(source_id: str, timestamp_us: int, text: str) -> dict:
        """
        Append a transcript segment for *source_id* at *timestamp_us*
        (Unix microseconds). ``source_id`` is any string — a real
        participant identity or an internal source name (e.g. ``"agent-vlm"``).

        Returns ``{"ok": true}`` on success, or an error dict if *text* is empty.
        """
        if not text.strip():
            return {"error": "text must not be empty"}
        store.append(source_id, timestamp_us, text)
        log.info("add_transcript  source=%r  ts=%d  %r", source_id, timestamp_us, text[:80])
        return {"ok": True}

    @mcp.tool()
    def list_sources() -> list[str]:
        """Return all source IDs that have at least one stored transcript."""
        return store.list_sources()

    @mcp.tool()
    def get_transcript_stats(source_id: str) -> dict:
        """
        Return summary statistics for *source_id*'s stored transcripts.

        Keys: source_id, count (utterances), total_chars,
              earliest_us (Unix µs), latest_us (Unix µs).
        Returns an error dict if no transcripts exist.
        """
        result = store.stats(source_id)
        if result is None:
            return {"error": f"No transcripts for {source_id!r}"}
        return result

    return mcp


def build_app(store: TranscriptStore):
    """Return the ASGI app serving the FastMCP HTTP transport at /mcp."""
    return build_mcp(store).http_app(path="/mcp")


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

    log.info("transcript-mcp-server  mcp=/mcp  port=%d  dir=%s",
             port, transcripts_dir)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
