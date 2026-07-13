# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only smoke tests for the semantic_slam in-process API.

These tests exercise the two pure helpers in ``semantic_slam.engine``
(``_rank_objects`` and ``_normalize_pose``) plus the real
``MapObjectList.compute_similarities`` cosine-similarity path. They construct no
GPU models and require no downloaded weights.

Run with:
    python -m pytest tests/test_semantic_slam_smoke.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
o3d = pytest.importorskip("open3d")
import torch.nn.functional as F

# Imported at module top on purpose: the whole point is that engine.py's module
# scope is GPU-/dependency-free so these helpers import without CUDA, open_clip
# or groundingdino installed.
from semantic_slam.engine import _normalize_pose, _rank_objects


class _FakeMap(list):
    """Minimal stand-in for MapObjectList.

    ``_rank_objects`` is duck-typed: it only needs ``len()``, ``[]`` indexing
    and a ``compute_similarities`` method. We use a fake here because the real
    ``slam.core.slam_classes`` cannot be imported standalone in a CPU-only env
    (``slam/__init__.py`` eagerly pulls in the GPU/external model stack; see the
    handoff notes). ``compute_similarities`` reproduces MapObjectList's exact
    cosine-similarity formula.
    """

    def compute_similarities(self, new_clip_ft):
        new_clip_ft = torch.as_tensor(new_clip_ft)
        clip_fts = torch.stack([o["clip_ft"] for o in self])
        return F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)


def _make_fake_object(clip_ft, center):
    """Build a minimal map object: a clip feature + a tiny pcd + an OBB."""
    pcd = o3d.geometry.PointCloud()
    pts = np.array(center, dtype=np.float64) + np.random.randn(8, 3) * 0.01
    pcd.points = o3d.utility.Vector3dVector(pts)
    # Explicit constructor (not create_from_points) -> deterministic, no
    # coplanar-degeneracy throw.
    bbox = o3d.geometry.OrientedBoundingBox(
        np.array(center, dtype=np.float64),
        np.eye(3),
        np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )
    return {
        "clip_ft": clip_ft,
        "class_name": ["item"],
        "class_id": [0],
        "pcd": pcd,
        "bbox": bbox,
    }


def _build_map():
    """A map with 3 objects whose clip features are e0, e1, e2."""
    objects = _FakeMap()
    for i, center in enumerate([(0, 0, 0), (1, 1, 1), (2, 2, 2)]):
        feat = torch.zeros(512)
        feat[i] = 1.0  # orthonormal basis vectors e0, e1, e2
        objects.append(_make_fake_object(feat, center))
    return objects


def test_query_ranking_picks_matching_object():
    """A query feature equal to e1 must rank object 1 first."""
    objects = _build_map()

    query_feat = torch.zeros(512)
    query_feat[1] = 1.0

    hits = _rank_objects(objects, query_feat, top_k=3)

    assert len(hits) == 3
    # Best hit corresponds to object 1: its centroid sits near (1, 1, 1).
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(hits[0]["centroid"], [1, 1, 1], atol=0.1)
    # The two orthogonal objects have ~0 cosine similarity.
    assert hits[1]["score"] == pytest.approx(0.0, abs=1e-5)
    assert hits[2]["score"] == pytest.approx(0.0, abs=1e-5)
    # Sorted descending.
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_query_result_schema():
    """Each hit exposes the documented keys with sane types."""
    objects = _build_map()
    query_feat = torch.zeros(512)
    query_feat[0] = 1.0

    hit = _rank_objects(objects, query_feat, top_k=1)[0]

    assert set(hit) == {
        "score",
        "class_name",
        "centroid",
        "num_points",
        "bbox_center",
        "bbox_extent",
    }
    assert hit["class_name"] == "item"
    assert hit["num_points"] == 8
    assert len(hit["centroid"]) == 3
    assert len(hit["bbox_center"]) == 3
    assert len(hit["bbox_extent"]) == 3


def test_rank_objects_top_k_clamped():
    """top_k larger than the map size returns one hit per object, no error."""
    objects = _build_map()
    query_feat = torch.zeros(512)
    query_feat[2] = 1.0

    hits = _rank_objects(objects, query_feat, top_k=10)
    assert len(hits) == 3


def test_rank_objects_empty_map():
    """An empty map yields no hits."""
    assert _rank_objects(_FakeMap(), torch.zeros(512), top_k=5) == []


def test_real_mapobjectlist_compute_similarities():
    """The real MapObjectList.compute_similarities matches our fake's formula.

    Skips cleanly when ``slam`` cannot be imported standalone (it eagerly pulls
    in an undeclared/uninstalled GPU dependency, ``gradslam``, in this env). In
    the full pipeline environment this test actually runs and pins the contract.
    """
    slam_classes = pytest.importorskip("slam.core.slam_classes")
    objects = slam_classes.MapObjectList()
    for i, center in enumerate([(0, 0, 0), (1, 1, 1), (2, 2, 2)]):
        feat = torch.zeros(512)
        feat[i] = 1.0
        objects.append(_make_fake_object(feat, center))

    q = torch.zeros(512)
    q[1] = 1.0
    sims = objects.compute_similarities(q)
    assert sims.shape == (3,)
    assert sims[1].item() == pytest.approx(1.0, abs=1e-5)
    assert sims[0].item() == pytest.approx(0.0, abs=1e-5)


def test_normalize_pose_from_4x4():
    """A 4x4 matrix is flattened row-major and prefixed with the frame number."""
    mat = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = _normalize_pose(mat, frame_number=7)

    assert out[0] == 7
    assert len(out) == 17
    # Row-major / C-order: matches server/main.py's replica producer.
    assert out[1:] == list(range(16))


def test_normalize_pose_from_flat_16():
    """A flat length-16 sequence passes through unchanged after the prefix."""
    flat = list(range(100, 116))
    out = _normalize_pose(flat, frame_number=3)

    assert out[0] == 3
    assert out[1:] == [float(x) for x in flat]


def test_normalize_pose_accepts_nested_list():
    """A 4x4 nested Python list is accepted, not just an ndarray."""
    mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    out = _normalize_pose(mat, frame_number=0)
    assert out == [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 3)),  # wrong matrix shape
        np.zeros(15),  # too short
        np.zeros(17),  # too long
        np.zeros((4, 4, 1)),  # too many dims
    ],
)
def test_normalize_pose_bad_shape_raises(bad):
    with pytest.raises(ValueError):
        _normalize_pose(bad, frame_number=0)
