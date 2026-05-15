# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online region tracker.

State machine per call:

* No regions exist yet              → seed region 0 from this descriptor.
* Best-match similarity ≥ threshold → snap to that region (running-mean
                                       update of its centroid); link from
                                       the previously-current region.
* Best-match similarity <  threshold → start (or continue) a "candidate"
                                       buffer.  After
                                       ``new_region_min_streak`` consecutive
                                       low-similarity frames whose pairwise
                                       similarity is high, commit them as
                                       a new region.  This kills spurious
                                       new-region creations caused by a
                                       single off-angle frame.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from .embedder import Embedder
from .regions  import Region, RegionStore


@dataclasses.dataclass(frozen=True)
class ProcessResult:
    """Return value of :func:`Tracker.process`."""
    state:         str                # "seeded" | "snapped" | "transitioned" | "pending_new" | "created"
    region_id:     int | None
    region_name:   str | None
    confidence:    float              # best cosine similarity vs. existing regions
    num_regions:   int
    transitioned_from: int | None
    ts_us:         int


class Tracker:
    def __init__(
        self,
        *,
        embedder:               Embedder,
        store:                  RegionStore,
        match_threshold:        float = 0.78,
        new_region_min_streak:  int   = 3,
        centroid_alpha:         float = 0.05,
    ) -> None:
        self._embedder              = embedder
        self._store                 = store
        self._match_threshold       = float(match_threshold)
        self._new_region_min_streak = max(1, int(new_region_min_streak))
        self._centroid_alpha        = float(centroid_alpha)

        # Region tracking state.
        self._current_region_id: int | None = None
        # Candidate buffer: descriptors that didn't match any region.  When
        # the buffer reaches `new_region_min_streak` AND its descriptors
        # cluster tightly (mean pairwise sim > threshold), commit a new
        # region with the mean centroid.
        self._candidate_buffer: list[np.ndarray] = []

    def process(self, image_rgb: np.ndarray, ts_us: int | None = None) -> ProcessResult:
        from loguru import logger
        if ts_us is None:
            ts_us = int(time.time() * 1_000_000)
        descriptor = self._embedder(image_rgb)

        # ── empty store: seed the first region immediately ───────────────
        if len(self._store) == 0:
            r = self._store.insert(descriptor, ts_us=ts_us)
            self._current_region_id = r.id
            self._candidate_buffer.clear()
            logger.info("space-mcp: seeded region {} (first frame)", r.id)
            return ProcessResult(
                state="seeded", region_id=r.id, region_name=r.name,
                confidence=1.0, num_regions=len(self._store),
                transitioned_from=None, ts_us=ts_us,
            )

        # ── match against existing regions ───────────────────────────────
        match = self._store.best_match(descriptor)
        assert match is not None    # guarded by len(self._store)==0 above
        best_id, best_sim = match

        if best_sim >= self._match_threshold:
            prev = self._current_region_id
            self._candidate_buffer.clear()
            self._store.update_centroid(
                best_id, descriptor,
                ts_us=ts_us, alpha=self._centroid_alpha,
            )
            transitioned = prev is not None and prev != best_id
            if transitioned:
                self._store.add_edge(prev, best_id)
            self._current_region_id = best_id
            r = self._store.get(best_id)
            return ProcessResult(
                state="transitioned" if transitioned else "snapped",
                region_id=best_id, region_name=r.name if r else None,
                confidence=best_sim, num_regions=len(self._store),
                transitioned_from=prev if transitioned else None, ts_us=ts_us,
            )

        # ── below threshold: candidate buffering ─────────────────────────
        self._candidate_buffer.append(descriptor)
        if len(self._candidate_buffer) < self._new_region_min_streak:
            return ProcessResult(
                state="pending_new", region_id=self._current_region_id,
                region_name=None, confidence=best_sim,
                num_regions=len(self._store),
                transitioned_from=None, ts_us=ts_us,
            )

        # Require the candidate buffer to be internally consistent — a
        # cluster of N similar low-match frames, not N wildly different
        # frames.  Otherwise we'd commit on every brief glance.
        buf = np.stack(self._candidate_buffer, axis=0)
        mean = buf.mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
        intra_sims = buf @ mean
        if float(intra_sims.min()) < self._match_threshold:
            # Not a tight cluster.  Drop the oldest sample, wait for more.
            self._candidate_buffer.pop(0)
            return ProcessResult(
                state="pending_new", region_id=self._current_region_id,
                region_name=None, confidence=best_sim,
                num_regions=len(self._store),
                transitioned_from=None, ts_us=ts_us,
            )

        # Commit the new region with the buffer's mean centroid.
        new_region = self._store.insert(mean.astype(np.float32), ts_us=ts_us)
        if self._current_region_id is not None:
            self._store.add_edge(self._current_region_id, new_region.id)
        prev = self._current_region_id
        self._current_region_id = new_region.id
        self._candidate_buffer.clear()
        logger.info(
            "space-mcp: created region {} from {} candidate frames (prev={}, sim={:.3f})",
            new_region.id, len(buf), prev, best_sim,
        )
        return ProcessResult(
            state="created", region_id=new_region.id, region_name=None,
            confidence=best_sim, num_regions=len(self._store),
            transitioned_from=prev, ts_us=ts_us,
        )

    # ── accessors ──────────────────────────────────────────────────────────

    @property
    def current_region_id(self) -> int | None:
        return self._current_region_id

    def reset(self) -> None:
        self._store.reset()
        self._current_region_id = None
        self._candidate_buffer.clear()
