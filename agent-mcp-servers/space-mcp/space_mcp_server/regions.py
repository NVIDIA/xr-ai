# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Region store + topological graph.

Each region is a node in the spatial memory: a centroid descriptor, an
optional human-readable name, a list of observed objects (filled in by
a separate object-recognition pipeline — currently a stub), and the set
of neighbour region ids the camera has been observed transitioning to.

Persistence is a single JSONL file (``regions.jsonl``) — atomic rewrite
on any structural change (rename, edge insertion, eviction, reset) so a
crash mid-write leaves the previous state intact.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import tempfile
import time
from typing import Iterable

import numpy as np


@dataclasses.dataclass
class Region:
    id:        int
    name:      str | None
    centroid:  np.ndarray         # (D,) float32, L2-normalized
    n_samples: int
    ts_first:  int                # unix microseconds
    ts_last:   int
    neighbors: set[int]
    objects:   list[dict]         # [{name, ts_last, frame_count}, ...]


class RegionStore:
    """In-memory list + JSONL persistence.  Single mutator at a time."""

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root
        self._regions: list[Region] = []
        self._next_id = 0
        self._root.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        jl = self._root / "regions.jsonl"
        if not jl.exists():
            return
        for line in jl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            r = Region(
                id        = int(row["id"]),
                name      = row.get("name"),
                centroid  = np.asarray(row["centroid"], dtype=np.float32),
                n_samples = int(row["n_samples"]),
                ts_first  = int(row["ts_first"]),
                ts_last   = int(row["ts_last"]),
                neighbors = set(int(x) for x in row.get("neighbors", [])),
                objects   = list(row.get("objects", [])),
            )
            self._regions.append(r)
            self._next_id = max(self._next_id, r.id + 1)

    def _flush(self) -> None:
        path = self._root / "regions.jsonl"
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in self._regions:
                    f.write(json.dumps({
                        "id":        r.id,
                        "name":      r.name,
                        "centroid":  r.centroid.tolist(),
                        "n_samples": r.n_samples,
                        "ts_first":  r.ts_first,
                        "ts_last":   r.ts_last,
                        "neighbors": sorted(r.neighbors),
                        "objects":   r.objects,
                    }) + "\n")
            os.replace(tmp, path)
        except Exception:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise

    # ── queries ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._regions)

    def all(self) -> list[Region]:
        return list(self._regions)

    def get(self, region_id: int) -> Region | None:
        for r in self._regions:
            if r.id == region_id:
                return r
        return None

    def best_match(self, descriptor: np.ndarray) -> tuple[int, float] | None:
        """Return ``(region_id, cosine_sim)`` for the best-matching region,
        or ``None`` if the store is empty.  Inputs are assumed already
        L2-normalized so dot product == cosine."""
        if not self._regions:
            return None
        cents = np.stack([r.centroid for r in self._regions], axis=0)
        sims = cents @ descriptor
        idx = int(np.argmax(sims))
        return self._regions[idx].id, float(sims[idx])

    def stats(self) -> dict:
        return {
            "map_dir":     str(self._root),
            "num_regions": len(self._regions),
            "num_edges":   sum(len(r.neighbors) for r in self._regions) // 2,
            "regions":     [
                {"id": r.id, "name": r.name,
                 "n_samples": r.n_samples,
                 "neighbors": sorted(r.neighbors)}
                for r in self._regions
            ],
        }

    # ── mutations ──────────────────────────────────────────────────────────

    def insert(self, descriptor: np.ndarray, *, ts_us: int) -> Region:
        r = Region(
            id=self._next_id, name=None,
            centroid=np.ascontiguousarray(descriptor, dtype=np.float32),
            n_samples=1, ts_first=int(ts_us), ts_last=int(ts_us),
            neighbors=set(), objects=[],
        )
        self._regions.append(r)
        self._next_id += 1
        self._flush()
        return r

    def update_centroid(
        self, region_id: int, descriptor: np.ndarray, *, ts_us: int, alpha: float,
    ) -> Region | None:
        """Weighted running average of the centroid.  Persists on every
        update so a long session doesn't lose drift if the server dies."""
        r = self.get(region_id)
        if r is None:
            return None
        new = (1.0 - alpha) * r.centroid + alpha * descriptor
        new = new / max(float(np.linalg.norm(new)), 1e-12)   # renormalize
        r.centroid  = new.astype(np.float32)
        r.n_samples += 1
        r.ts_last   = int(ts_us)
        self._flush()
        return r

    def add_edge(self, a: int, b: int) -> bool:
        """Mark regions ``a`` and ``b`` as topological neighbours.  Returns
        True iff a new edge was added."""
        if a == b:
            return False
        ra, rb = self.get(a), self.get(b)
        if ra is None or rb is None:
            return False
        if b in ra.neighbors and a in rb.neighbors:
            return False
        ra.neighbors.add(b)
        rb.neighbors.add(a)
        self._flush()
        return True

    def rename(self, region_id: int, name: str | None) -> Region | None:
        r = self.get(region_id)
        if r is None:
            return None
        r.name = name if name else None
        self._flush()
        return r

    def reset(self) -> None:
        self._regions.clear()
        self._next_id = 0
        jl = self._root / "regions.jsonl"
        if jl.exists():
            jl.unlink()
