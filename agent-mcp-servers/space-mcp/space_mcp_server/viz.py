# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rerun visualization for the topological place memory.

Layout: every region is placed deterministically by its id on a Fibonacci
spiral in the X-Z plane (Y up), so positions are stable across runs and
don't shift when a new region is added.  Edges in the region graph are
drawn as thick line strips between region positions.  The current region
is highlighted with a "you-are-here" marker plus the latest RGB image
logged underneath it.

Entity tree::

    /space/origin                   axes anchor
    /space/regions/r{ID}            Transform3D at the layout position
            ...     /marker         Points3D (label = region name or "r{ID}")
            ...     /objects        TextDocument listing the object catalog
            ...     /thumb          Image — the keyframe-ish view that
                                     seeded the region, when available
    /space/edges                    LineStrips3D over all neighbour pairs
    /space/current                  marker at the currently-occupied region
            ...     /image          live RGB frame logged each call
            ...     /label          TextDocument summarising the current
                                     region + neighbours + recent objects
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .regions import Region
from .tracker import ProcessResult


class VizSink(Protocol):
    def on_load(self, regions: list[Region]) -> None: ...
    def on_process(
        self,
        result:     ProcessResult,
        regions:    list[Region],
        image_rgb:  np.ndarray,
    ) -> None: ...
    def on_objects(self, region: Region) -> None: ...


def _region_position(region_id: int) -> tuple[float, float, float]:
    """Deterministic Fibonacci-spiral layout, X-Z plane (Y up).  Stable per
    id and independent of the current region count, so adding a new
    region never shifts an existing one."""
    angle  = region_id * 2.0 * math.pi * 0.6180339887
    radius = math.sqrt(region_id + 1) * 2.0
    return (radius * math.cos(angle), 0.0, radius * math.sin(angle))


_CHAIN_COLOR    = [120, 120, 130]
_REGION_COLOR   = [180, 180, 200]
_CURRENT_COLOR  = [255, 215,  60]


class RerunSink:
    """Streams topological map updates into a Rerun viewer."""

    def __init__(
        self, *,
        application_id: str = "space-mcp",
        addr:           str = "127.0.0.1:9876",
    ) -> None:
        self._app_id = application_id
        self._addr   = addr
        self._rr     = None

    def _ensure_connected(self) -> None:
        if self._rr is not None:
            return
        import rerun as rr
        from loguru import logger
        url = self._addr if "://" in self._addr else f"rerun+http://{self._addr}/proxy"
        logger.info("Rerun: connecting to {} (app_id={!r})", url, self._app_id)
        rr.init(self._app_id)
        connect_kwargs = {}
        try:
            from rerun.blueprint import Blueprint, Spatial3DView
            connect_kwargs["default_blueprint"] = Blueprint(
                Spatial3DView(origin="/space", name="space-mcp topological map"),
                collapse_panels=True,
            )
        except Exception as exc:
            logger.debug("Rerun: skipping default_blueprint ({})", exc)
        rr.connect_grpc(url, **connect_kwargs)
        # Origin axes — orients the viewer for the spiral layout.
        rr.log("/space/origin",
               rr.Transform3D(translation=[0.0, 0.0, 0.0]),
               static=True)
        logger.info("Rerun: connected; topological map sink ready")
        self._rr = rr

    # ── lifecycle ──────────────────────────────────────────────────────────

    def on_load(self, regions: list[Region]) -> None:
        self._ensure_connected()
        for r in regions:
            self._log_region(r, static=True)
        self._log_edges(regions, static=True)

    def on_process(
        self,
        result:    ProcessResult,
        regions:   list[Region],
        image_rgb: np.ndarray,
    ) -> None:
        self._ensure_connected()
        rr = self._rr
        rr.set_time("frame_time", timestamp=result.ts_us / 1_000_000.0)

        # State-change refreshes: a new region means a new node + edges to
        # re-log.  A transition redraws to update the "current" marker only.
        if result.state in ("seeded", "created"):
            new_r = next((r for r in regions if r.id == result.region_id), None)
            if new_r is not None:
                self._log_region(new_r, static=False)
                self._log_edges(regions, static=False)
        elif result.state == "transitioned":
            # Edges may have gained one — refresh.
            self._log_edges(regions, static=False)

        # Always update the "you are here" marker + the live image.
        if result.region_id is None:
            rr.log("/space/current", rr.Clear(recursive=True))
            return
        cur = next((r for r in regions if r.id == result.region_id), None)
        if cur is None:
            return
        pos = _region_position(cur.id)
        rr.log("/space/current",
               rr.Transform3D(translation=pos))
        rr.log("/space/current/marker",
               rr.Points3D(
                   positions=np.array([pos], dtype=np.float32),
                   radii=np.array([0.45], dtype=np.float32),
                   colors=np.array([_CURRENT_COLOR], dtype=np.uint8),
                   labels=[cur.name or f"r{cur.id}"],
               ))
        rr.log("/space/current/image", rr.Image(image_rgb))
        rr.log("/space/current/label",
               rr.TextDocument(self._region_summary(cur, regions)))

    def on_objects(self, region: Region) -> None:
        self._ensure_connected()
        self._log_region(region, static=False)

    # ── helpers ────────────────────────────────────────────────────────────

    def _log_region(self, region: Region, *, static: bool) -> None:
        rr = self._rr
        pos  = _region_position(region.id)
        path = f"/space/regions/r{region.id}"
        rr.log(path,
               rr.Transform3D(translation=pos), static=static)
        rr.log(f"{path}/marker",
               rr.Points3D(
                   positions=np.array([pos], dtype=np.float32),
                   radii=np.array([0.30], dtype=np.float32),
                   colors=np.array([_REGION_COLOR], dtype=np.uint8),
                   labels=[region.name or f"r{region.id}"],
               ),
               static=static)
        if region.objects:
            rr.log(f"{path}/objects",
                   rr.TextDocument(
                       f"# r{region.id}" + (f" — {region.name}" if region.name else "") +
                       "\n\n" + "\n".join(
                           f"- {o.get('name')} (×{o.get('frame_count', 1)})"
                           for o in region.objects
                       )
                   ),
                   static=static)

    def _log_edges(self, regions: list[Region], *, static: bool) -> None:
        seen: set[tuple[int, int]] = set()
        strips: list[np.ndarray] = []
        for r in regions:
            for n in r.neighbors:
                key = (min(r.id, n), max(r.id, n))
                if key in seen:
                    continue
                seen.add(key)
                p1 = np.asarray(_region_position(r.id), dtype=np.float32)
                p2 = np.asarray(_region_position(n),    dtype=np.float32)
                strips.append(np.stack([p1, p2], axis=0))
        if not strips:
            return
        self._rr.log(
            "/space/edges",
            self._rr.LineStrips3D(
                strips,
                colors=np.tile(_CHAIN_COLOR, (len(strips), 1)).astype(np.uint8),
                radii=np.full((len(strips),), 0.04, dtype=np.float32),
            ),
            static=static,
        )

    @staticmethod
    def _region_summary(cur: Region, regions: list[Region]) -> str:
        # Build a compact markdown block that fits the side panel.
        by_id = {r.id: r for r in regions}
        neighbours = ", ".join(
            (by_id[n].name or f"r{n}") for n in sorted(cur.neighbors) if n in by_id
        ) or "—"
        objects = ", ".join(o.get("name", "?") for o in cur.objects[:12]) or "—"
        return (
            f"## r{cur.id}" + (f" — {cur.name}" if cur.name else "") + "\n\n"
            f"**samples:** {cur.n_samples}\n\n"
            f"**neighbours:** {neighbours}\n\n"
            f"**objects:** {objects}\n"
        )
