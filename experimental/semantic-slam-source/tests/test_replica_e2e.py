# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the SemanticSLAM API against a real Replica scene.

Exercises the full real pipeline: build the engine, push a handful of
RGB/depth/pose frames from Replica room0, then run text queries against the
resulting semantic map.

Requires the full model stack (run scripts/setup_env.sh), a CUDA GPU, and the
Replica dataset. Skips cleanly when any of those is absent, so it is safe in CI.

Run directly for verbose output:
    REPLICA_ROOT=/data/replica/Replica \
      .venv/bin/python tests/test_replica_e2e.py
Or via pytest:
    .venv/bin/python -m pytest tests/test_replica_e2e.py -q -s
"""

import os
from pathlib import Path

import numpy as np
import pytest

REPLICA_ROOT = Path(os.environ.get("REPLICA_ROOT", "/data/replica/Replica"))
SCENE = os.environ.get("REPLICA_SCENE", "room0")
CONFIG = Path(__file__).resolve().parent.parent / "config" / "semantic_slam_test.yaml"
NUM_FRAMES = int(os.environ.get("E2E_NUM_FRAMES", "4"))
STRIDE = int(os.environ.get("E2E_STRIDE", "50"))


def _require_environment():
    """Skip the test unless GPU + model stack + dataset are all available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for the end-to-end pipeline")
    for mod in ("segment_anything", "pytorch3d", "gradslam", "open_clip"):
        pytest.importorskip(mod)
    scene_dir = REPLICA_ROOT / SCENE
    if not (scene_dir / "results").is_dir() or not (scene_dir / "traj.txt").is_file():
        pytest.skip(f"Replica scene not found at {scene_dir}")


def _load_frames(n, stride):
    """Yield (rgb, depth, pose4x4) tuples from Replica room0."""
    from PIL import Image

    results = REPLICA_ROOT / SCENE / "results"
    poses = np.loadtxt(REPLICA_ROOT / SCENE / "traj.txt")  # (N, 16)
    for i in range(n):
        idx = i * stride
        rgb_path = results / f"frame{idx:06d}.jpg"
        depth_path = results / f"depth{idx:06d}.png"
        if not rgb_path.exists() or not depth_path.exists():
            break
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.asarray(Image.open(depth_path))
        pose = poses[idx].reshape(4, 4)
        yield idx, rgb, depth, pose


def _build_engine():
    from config.settings import Config
    from semantic_slam import SemanticSLAM

    config = Config.from_yaml(str(CONFIG))
    return SemanticSLAM(
        dataset_type="replica",
        scene_name=SCENE,
        config=config,
        use_detector=False,
        device="cuda:0",
    )


def test_replica_push_and_query():
    _require_environment()
    slam = _build_engine()

    pushed = 0
    for idx, rgb, depth, pose in _load_frames(NUM_FRAMES, STRIDE):
        n_objects = slam.push(rgb, depth, pose, frame_number=idx)
        pushed += 1
        print(f"  frame {idx}: map now has {n_objects} objects")

    assert pushed > 0, "no frames were loaded from the Replica scene"
    assert len(slam.objects) > 0, "pipeline produced an empty map"

    hits = slam.query("a chair", top_k=5)
    print(f"  query 'a chair' -> {len(hits)} hits")
    assert isinstance(hits, list) and len(hits) > 0
    for h in hits:
        assert {"score", "centroid"} <= set(h)
        assert np.isfinite(h["score"])
        assert len(h["centroid"]) == 3
    # results are ranked best-first
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    # Standalone runner with human-readable output.
    _require_environment.__wrapped__ if hasattr(_require_environment, "__wrapped__") else None
    print(f"Replica: {REPLICA_ROOT/SCENE}  config: {CONFIG}")
    slam = _build_engine()
    for idx, rgb, depth, pose in _load_frames(NUM_FRAMES, STRIDE):
        n = slam.push(rgb, depth, pose, frame_number=idx)
        print(f"pushed frame {idx}: {n} objects")
    print(f"\nFinal map: {len(slam.objects)} objects")
    for text in ("a chair", "a sofa", "a table", "a window"):
        hits = slam.query(text, top_k=3)
        print(f"\nquery {text!r}:")
        for h in hits:
            c = h["centroid"]
            print(f"  score={h['score']:.3f}  centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  "
                  f"class={h.get('class_name')}  pts={h.get('num_points')}")
