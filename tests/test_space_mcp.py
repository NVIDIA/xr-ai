# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for space-mcp's topological place memory.

Heavy DINOv2 inference is out of scope — covered by a GPU-marked test
elsewhere.  Here we use a deterministic fake embedder that returns
preset descriptors, which lets us exercise the Tracker state machine
plus the RegionStore's persistence path on CPU in milliseconds.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from space_mcp_server.embedder import Embedder
from space_mcp_server.regions  import RegionStore
from space_mcp_server.tracker  import Tracker


class _ScriptedEmbedder:
    """Returns successive L2-normalized descriptors from a fixed script.

    Implements the :class:`Embedder` protocol so the Tracker accepts it.
    Anything beyond the script's length re-emits the last descriptor.
    """
    embedding_dim = 4

    def __init__(self, descriptors: list[np.ndarray]) -> None:
        norm = lambda v: v / max(np.linalg.norm(v), 1e-12)
        self._script = [norm(np.asarray(d, dtype=np.float32)) for d in descriptors]
        self._i = 0

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        i = min(self._i, len(self._script) - 1)
        self._i += 1
        return self._script[i].copy()


# Conventional dummy image — the fake embedder ignores it but the
# Tracker pipeline still requires a numpy array shape.
_DUMMY = np.zeros((4, 4, 3), dtype=np.uint8)


def _tracker(
    tmp_path: pathlib.Path,
    descriptors: list[list[float]],
    **kwargs,
) -> Tracker:
    return Tracker(
        embedder=_ScriptedEmbedder([np.asarray(d) for d in descriptors]),
        store=RegionStore(tmp_path),
        **kwargs,
    )


# ── basic state machine ──────────────────────────────────────────────────

def test_first_frame_seeds_region_zero(tmp_path):
    t = _tracker(tmp_path, [[1, 0, 0, 0]])
    r = t.process(_DUMMY, ts_us=1)
    assert r.state == "seeded"
    assert r.region_id == 0
    assert r.num_regions == 1


def test_same_view_snaps_to_existing_region(tmp_path):
    # Two identical frames → second snaps to region 0, no transition.
    t = _tracker(tmp_path, [[1, 0, 0, 0], [1, 0, 0, 0]])
    r1 = t.process(_DUMMY, ts_us=1)
    r2 = t.process(_DUMMY, ts_us=2)
    assert r1.region_id == 0
    assert r2.state == "snapped"
    assert r2.region_id == 0
    assert r2.transitioned_from is None
    assert r2.num_regions == 1


def test_orthogonal_view_after_streak_creates_new_region(tmp_path):
    # Region 0 from frame 1; then three orthogonal frames cluster into
    # a new region.  Edge between region 0 and the new region is added.
    t = _tracker(
        tmp_path,
        [[1, 0, 0, 0],          # seeds region 0
         [0, 1, 0, 0],          # pending_new
         [0, 1, 0, 0.01],       # pending_new (tight cluster)
         [0, 1, 0, -0.01]],     # created
        new_region_min_streak=3, match_threshold=0.78,
    )
    r0 = t.process(_DUMMY, ts_us=1)
    assert r0.state == "seeded"
    r1 = t.process(_DUMMY, ts_us=2)
    assert r1.state == "pending_new"
    assert r1.region_id == 0          # still in region 0 as far as the agent knows
    r2 = t.process(_DUMMY, ts_us=3)
    assert r2.state == "pending_new"
    r3 = t.process(_DUMMY, ts_us=4)
    assert r3.state == "created"
    assert r3.region_id == 1
    assert r3.transitioned_from == 0
    # Edge added between 0 and 1.
    r0_full = t._store.get(0)
    r1_full = t._store.get(1)
    assert r1_full.id in r0_full.neighbors
    assert r0_full.id in r1_full.neighbors


def test_transition_back_records_edge_once(tmp_path):
    # Seed region 0, create region 1 via a streak, walk back to region 0:
    # second transition adds no duplicate edge.
    descs = [
        [1, 0, 0, 0],   # seed r0
        [0, 1, 0, 0],   # pending_new
        [0, 1, 0, 0],   # pending_new
        [0, 1, 0, 0],   # created r1
        [1, 0, 0, 0],   # back to r0 → transitioned
        [1, 0, 0, 0],   # snapped (no transition)
    ]
    t = _tracker(tmp_path, descs, new_region_min_streak=3, match_threshold=0.78)
    states = [t.process(_DUMMY, ts_us=i).state for i in range(len(descs))]
    assert states == [
        "seeded", "pending_new", "pending_new", "created",
        "transitioned", "snapped",
    ]
    # Single edge {0,1} regardless of how many times we move between them.
    r0 = t._store.get(0)
    r1 = t._store.get(1)
    assert r0.neighbors == {1}
    assert r1.neighbors == {0}


def test_unclustered_buffer_does_not_create_region(tmp_path):
    # Three low-similarity frames that *don't* cluster together → no new
    # region (false-positive guard).
    descs = [
        [1, 0, 0, 0],            # seed r0
        [0, 1, 0, 0],            # pending_new
        [0, 0, 1, 0],            # also low sim to r0 *and* to prev candidate
        [0, 0, 0, 1],            # likewise
        [0, 0, 0, 1],            # likewise — should still not create
    ]
    t = _tracker(tmp_path, descs, new_region_min_streak=3, match_threshold=0.78)
    results = [t.process(_DUMMY, ts_us=i) for i in range(len(descs))]
    assert results[0].state == "seeded"
    for r in results[1:]:
        assert r.state == "pending_new", f"unexpected state {r.state}"
    assert len(t._store) == 1


# ── persistence ──────────────────────────────────────────────────────────

def test_regions_persist_across_restart(tmp_path):
    # Build two regions, then drop the store and reload.
    descs = [
        [1, 0, 0, 0],
        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    t = _tracker(tmp_path, descs, new_region_min_streak=3, match_threshold=0.78)
    for i in range(len(descs)):
        t.process(_DUMMY, ts_us=i)

    n_before = len(t._store)
    assert n_before == 2

    reloaded = RegionStore(tmp_path)
    assert len(reloaded) == 2
    r0 = reloaded.get(0)
    r1 = reloaded.get(1)
    assert r0 is not None and r1 is not None
    assert r0.neighbors == {1}
    assert r1.neighbors == {0}
    # Centroid is still ~unit-norm after a round trip.
    np.testing.assert_allclose(np.linalg.norm(r0.centroid), 1.0, atol=1e-3)


def test_label_region_persists(tmp_path):
    store = RegionStore(tmp_path)
    store.insert(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), ts_us=0)
    store.rename(0, "kitchen")

    reloaded = RegionStore(tmp_path)
    assert reloaded.get(0).name == "kitchen"

    # Clearing the name also persists.
    reloaded.rename(0, None)
    again = RegionStore(tmp_path)
    assert again.get(0).name is None


def test_reset_wipes_store(tmp_path):
    store = RegionStore(tmp_path)
    store.insert(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), ts_us=0)
    assert len(store) == 1
    store.reset()
    assert len(store) == 0
    assert len(RegionStore(tmp_path)) == 0


def test_remember_objects_merges_and_persists(tmp_path):
    store = RegionStore(tmp_path)
    store.insert(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), ts_us=0)
    store.remember_objects(0, ["sofa", "lamp"], ts_us=100)
    store.remember_objects(0, ["LAMP", "Plant"], ts_us=200)   # case-insensitive merge

    r = RegionStore(tmp_path).get(0)
    names = {o["name"]: o for o in r.objects}
    assert set(names) == {"sofa", "lamp", "plant"}
    # "lamp" was seen twice (case-insensitive), so its count bumped.
    assert names["lamp"]["frame_count"] == 2
    assert names["sofa"]["frame_count"] == 1
    assert names["lamp"]["ts_last"] == 200
