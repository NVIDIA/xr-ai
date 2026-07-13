#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual validation of the SemanticSLAM API on a real Replica scene.

Builds a map by pushing a few RGB-D frames, then writes images you can eyeball:

  outputs/semantic_slam_viz/
    frame_XXXXXX_seg.png     SAM masks overlaid on the input RGB (per frame)
    map_bev.png              bird's-eye scatter of all object centroids
    query_<text>.png         BEV map with the top-k hits for a text query ringed

Run (headless-safe, matplotlib Agg):
    REPLICA_ROOT=/data/replica/Replica \
    GSA_PATH=$PWD/external/Grounded-Segment-Anything \
    HF_HOME=$HOME/.cache/huggingface \
    SEMANTIC_SLAM_AUTO_SETUP=0 \
      .venv/bin/python scripts/visualize_e2e.py
"""

import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLICA_ROOT = Path(os.environ.get("REPLICA_ROOT", "/data/replica/Replica"))
SCENE = os.environ.get("REPLICA_SCENE", "room0")
CONFIG = REPO_ROOT / "config" / "semantic_slam_test.yaml"
OUT = Path(os.environ.get("VIZ_OUT", REPO_ROOT / "outputs" / "semantic_slam_viz"))
NUM_FRAMES = int(os.environ.get("E2E_NUM_FRAMES", "4"))
STRIDE = int(os.environ.get("E2E_STRIDE", "50"))
QUERIES = ["a chair", "a sofa", "a table", "a window"]


def _load_frames(n, stride):
    from PIL import Image

    results = REPLICA_ROOT / SCENE / "results"
    poses = np.loadtxt(REPLICA_ROOT / SCENE / "traj.txt")
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


def _overlay_masks(rgb, masks, out_path):
    """Alpha-blend per-instance SAM masks over the RGB frame and save."""
    canvas = rgb.astype(np.float32).copy()
    if masks is not None and len(masks):
        rng = np.random.default_rng(0)
        colors = rng.integers(60, 255, size=(len(masks), 3)).astype(np.float32)
        for m, c in zip(masks, colors):
            m = np.asarray(m).astype(bool)
            if m.shape != rgb.shape[:2]:
                continue
            canvas[m] = 0.5 * canvas[m] + 0.5 * c
    plt.figure(figsize=(10, 6))
    plt.imshow(canvas.astype(np.uint8))
    n = 0 if masks is None else len(masks)
    plt.title(f"{out_path.stem} — {n} SAM masks")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def _centroids_and_sizes(objects):
    cents, sizes = [], []
    for obj in objects:
        pts = np.asarray(obj["pcd"].points)
        if len(pts) == 0:
            cents.append([0, 0, 0]); sizes.append(1); continue
        cents.append(pts.mean(axis=0))
        sizes.append(len(pts))
    return np.asarray(cents), np.asarray(sizes)


def _bev(cents, sizes, out_path, title, highlight=None, scores=None):
    """Bird's-eye (x vs z) scatter of object centroids; ring highlighted ones."""
    if len(cents) == 0:
        return
    s = 20 + 180 * (sizes / max(sizes.max(), 1))
    plt.figure(figsize=(8, 8))
    plt.scatter(cents[:, 0], cents[:, 2], s=s, c="lightgray", edgecolors="gray", alpha=0.8)
    if highlight is not None:
        hc = cents[highlight]
        plt.scatter(hc[:, 0], hc[:, 2], s=s[highlight] + 40, facecolors="none",
                    edgecolors="crimson", linewidths=2.5, zorder=3)
        for rank, idx in enumerate(highlight):
            lbl = f"#{rank+1}"
            if scores is not None:
                lbl += f" {scores[rank]:.2f}"
            plt.annotate(lbl, (cents[idx, 0], cents[idx, 2]), color="crimson",
                         fontsize=9, weight="bold",
                         xytext=(4, 4), textcoords="offset points")
    plt.xlabel("x (m)"); plt.ylabel("z (m)")
    plt.title(title); plt.gca().set_aspect("equal", "box"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close()


def main():
    from config.settings import Config
    from semantic_slam import SemanticSLAM
    from semantic_slam.engine import _rank_objects
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing images to {OUT}")

    slam = SemanticSLAM(dataset_type="replica", scene_name=SCENE,
                        config=Config.from_yaml(str(CONFIG)),
                        use_detector=False, device="cuda:0")

    for idx, rgb, depth, pose in _load_frames(NUM_FRAMES, STRIDE):
        # reuse the engine's own SAM model for the overlay
        masks, _, _ = slam.segmentation_model.run_segmentation(rgb, None)
        _overlay_masks(rgb, masks, OUT / f"frame_{idx:06d}_seg.png")
        n = slam.push(rgb, depth, pose, frame_number=idx)
        print(f"  frame {idx}: {len(masks) if masks is not None else 0} masks, map now {n} objects")

    cents, sizes = _centroids_and_sizes(slam.objects)
    _bev(cents, sizes, OUT / "map_bev.png", f"{SCENE}: {len(slam.objects)} mapped objects (BEV)")

    # one BEV per query, top-3 hits ringed
    for text in QUERIES:
        hits = slam.query(text, top_k=3)
        if not hits:
            continue
        # map hit centroids back to indices for highlighting
        scores = [h["score"] for h in hits]
        hit_cents = np.asarray([h["centroid"] for h in hits])
        idxs = [int(np.argmin(np.linalg.norm(cents - hc, axis=1))) for hc in hit_cents]
        safe = text.replace(" ", "_")
        _bev(cents, sizes, OUT / f"query_{safe}.png",
             f"query: {text!r} — top-3 ringed", highlight=idxs, scores=scores)
        print(f"  query {text!r}: top score {scores[0]:.3f}")

    print(f"\nDone. {len(list(OUT.glob('*.png')))} images in {OUT}")


if __name__ == "__main__":
    main()
