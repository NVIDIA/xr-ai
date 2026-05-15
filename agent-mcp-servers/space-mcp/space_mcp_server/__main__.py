# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
space-mcp server.

Topological place memory: each frame is embedded with DINOv2, matched
against the centroid of every known region by cosine similarity, and
either snapped to the best match (above ``match_threshold``) or seeded
as a new region.  Region transitions add edges in a persistent topological
graph.  This server intentionally answers the *coarse* localization
question — "which place am I in" rather than "where am I in metres" —
and is robust to depth / FOV pathologies that break metric pose pipelines.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  process_frame(image_path, timestamp_us=0) → dict
      Embed the frame, update the region map, return the current
      region's id + name + confidence.  This is the only model-running
      tool; everything else is a pure read against in-memory state.

  where_am_i() → dict
      Current region id, name, neighbours, recent objects (when wired).

  list_regions() → list[dict]
      One entry per region: id, name, n_samples, neighbours.

  describe_region(region_id) → dict
      Full record for one region.

  label_region(region_id, name) → dict
      Set a human-friendly name.  Persists to disk.

  reset_map() → dict
      Wipe the entire region store + topology.

Config (space_mcp_server.yaml)
───────────────────────────────
    map_dir:               /tmp/xr-ai/space-map
    device:                auto
    dinov2_model:          facebook/dinov2-small
    match_threshold:       0.78
    new_region_min_streak: 3
    centroid_alpha:        0.05
    host:                  0.0.0.0
    port:                  8245
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib

import numpy as np
import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from PIL import Image

from xr_ai_logging import setup_logging

from .embedder import DinoV2Embedder
from .regions  import RegionStore
from .tracker  import Tracker


def _load_image_rgb(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def build_mcp(tracker: Tracker, store: RegionStore) -> FastMCP:
    mcp = FastMCP("space-mcp")

    @mcp.tool()
    def process_frame(image_path: str, timestamp_us: int = 0) -> dict:
        """Process one frame, update the region map, return the inferred
        region.

        Returns
        -------
        dict
            state         : "seeded" | "snapped" | "transitioned"
                            | "pending_new" | "created"
            region_id     : int or None (None during pending_new)
            region_name   : str or None
            confidence    : best cosine similarity vs. existing regions
            num_regions   : map size after this call
            transitioned_from : previous region id when state == "transitioned"
                                or "created", else None
            ts_us         : echo of the timestamp used
        """
        path = pathlib.Path(image_path)
        if not path.exists():
            return {"error": f"image not found: {image_path!r}"}
        try:
            rgb = _load_image_rgb(path)
        except Exception as exc:
            return {"error": f"failed to load image: {exc}"}
        try:
            r = tracker.process(rgb, ts_us=timestamp_us or None)
        except Exception as exc:
            logger.exception("process_frame failed for {}", image_path)
            return {"error": f"region tracking failed: {exc}"}
        logger.info(
            "process_frame  state={}  region={}  conf={:.3f}  regions={}",
            r.state, r.region_id, r.confidence, r.num_regions,
        )
        return {
            "state":             r.state,
            "region_id":         r.region_id,
            "region_name":       r.region_name,
            "confidence":        r.confidence,
            "num_regions":       r.num_regions,
            "transitioned_from": r.transitioned_from,
            "ts_us":             r.ts_us,
        }

    @mcp.tool()
    def where_am_i() -> dict:
        """Return the agent's current region.  Cheap; no model inference."""
        rid = tracker.current_region_id
        if rid is None:
            return {"region_id": None, "message": "no frames processed yet"}
        r = store.get(rid)
        if r is None:
            return {"region_id": None, "message": "current region was evicted"}
        return {
            "region_id":  r.id,
            "name":       r.name,
            "n_samples":  r.n_samples,
            "neighbors":  sorted(r.neighbors),
            "objects":    r.objects,
        }

    @mcp.tool()
    def list_regions() -> dict:
        """Return every region in the map (id, name, sample count, neighbours).
        Pair with describe_region for full detail on one entry."""
        return store.stats()

    @mcp.tool()
    def describe_region(region_id: int) -> dict:
        """Full record for one region: name, sample count, first/last seen
        timestamps, neighbouring region ids, and the object catalog (when
        populated)."""
        r = store.get(int(region_id))
        if r is None:
            return {"error": f"no region with id {region_id}"}
        return {
            "id":        r.id,
            "name":      r.name,
            "n_samples": r.n_samples,
            "ts_first":  r.ts_first,
            "ts_last":   r.ts_last,
            "neighbors": sorted(r.neighbors),
            "objects":   r.objects,
        }

    @mcp.tool()
    def label_region(region_id: int, name: str) -> dict:
        """Set a human-friendly name for a region (e.g. "kitchen").  Pass
        an empty string to clear the name."""
        r = store.rename(int(region_id), name or None)
        if r is None:
            return {"error": f"no region with id {region_id}"}
        return {"id": r.id, "name": r.name}

    @mcp.tool()
    def reset_map() -> dict:
        """Wipe the entire region store + topology.  Next process_frame
        call seeds region 0 again."""
        tracker.reset()
        logger.warning("space-mcp: map wiped")
        return {"ok": True, "num_regions": 0}

    return mcp


def build_app(tracker: Tracker, store: RegionStore):
    return build_mcp(tracker, store).http_app(path="/mcp")


async def _serve(cfg: dict, ready_file: pathlib.Path | None) -> None:
    map_dir = pathlib.Path(
        cfg.get("map_dir", "/tmp/xr-ai/space-map")
    ).expanduser()
    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8245))

    store    = RegionStore(map_dir)
    embedder = DinoV2Embedder(
        model_name=cfg.get("dinov2_model", "facebook/dinov2-small"),
        device=cfg.get("device", "auto"),
    )
    tracker  = Tracker(
        embedder=embedder, store=store,
        match_threshold      =float(cfg.get("match_threshold",      0.78)),
        new_region_min_streak=int  (cfg.get("new_region_min_streak", 3)),
        centroid_alpha       =float(cfg.get("centroid_alpha",       0.05)),
    )

    app = build_app(tracker, store)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info(
        "space-mcp-server  map_dir={} (regions={})  port={}",
        map_dir, len(store), port,
    )
    if ready_file:
        ready_file.touch()
    await server.serve()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    setup_logging("space-mcp")
    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
