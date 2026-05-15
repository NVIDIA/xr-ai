# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose graph optimization wrapper around GTSAM.

Nodes are keyframe ids; edges are :class:`PoseEdge` instances carrying a
relative SE(3) transform between two keyframes and an inlier count used as
information weighting.  The origin keyframe (lowest id) is anchored with a
hard prior so the gauge is fixed.

Calling :func:`PoseGraph.optimize` returns a ``{kf_id: 4x4}`` dict of
post-optimization poses; the caller is responsible for writing them back
to its keyframe store.  We never touch keyframe.pts3d / kf.kp / kf.image —
those are in the keyframe's local frame and are unaffected by changes to
the keyframe's world pose.

GTSAM (BSD-3) is required; the operator-facing import error is loud and
explicit if it's missing rather than silently degrading.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import time
from typing import Iterable

import numpy as np


@dataclasses.dataclass(frozen=True)
class PoseEdge:
    """One pose-graph constraint.

    ``rel_pose`` is the SE(3) transform such that
    ``T_world_dst ≈ T_world_src @ rel_pose``.
    ``inliers`` weights the observation: higher = more confident.
    ``kind`` is "chain" for sequential adjacency or "loop" for loop closures
    — purely informational, doesn't change the math.
    """
    src_id:    int
    dst_id:    int
    rel_pose:  np.ndarray   # 4x4
    inliers:   int
    kind:      str = "chain"


class PoseGraph:
    """Persistent list of pose-graph edges with a GTSAM-backed optimizer."""

    def __init__(self, edges_path: pathlib.Path) -> None:
        self._path  = edges_path
        self._edges: list[PoseEdge] = []
        if self._path.exists():
            self._load()

    # ── persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self._edges.append(PoseEdge(
                src_id=int(row["src_id"]),
                dst_id=int(row["dst_id"]),
                rel_pose=np.asarray(row["rel_pose"], dtype=np.float64).reshape(4, 4),
                inliers=int(row.get("inliers", 0)),
                kind=str(row.get("kind", "chain")),
            ))

    def _append_to_disk(self, edge: PoseEdge) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "src_id":   edge.src_id,
                "dst_id":   edge.dst_id,
                "rel_pose": edge.rel_pose.tolist(),
                "inliers":  edge.inliers,
                "kind":     edge.kind,
            }) + "\n")

    # ── mutations ───────────────────────────────────────────────────────────

    def add(self, edge: PoseEdge) -> None:
        self._edges.append(edge)
        self._append_to_disk(edge)

    def clear(self) -> None:
        self._edges.clear()
        if self._path.exists():
            self._path.unlink()

    def drop_edges_touching(self, kf_id: int) -> None:
        """Remove every edge that mentions ``kf_id`` (used on FIFO eviction)."""
        kept = [e for e in self._edges if e.src_id != kf_id and e.dst_id != kf_id]
        if len(kept) == len(self._edges):
            return
        self._edges = kept
        # Atomic rewrite: tmp + replace so a crash doesn't leave a torn file.
        tmp = self._path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in self._edges:
                f.write(json.dumps({
                    "src_id": e.src_id, "dst_id": e.dst_id,
                    "rel_pose": e.rel_pose.tolist(),
                    "inliers": e.inliers, "kind": e.kind,
                }) + "\n")
        os.replace(tmp, self._path)

    # ── accessors ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._edges)

    def edges(self) -> list[PoseEdge]:
        return list(self._edges)

    def num_loops(self) -> int:
        return sum(1 for e in self._edges if e.kind == "loop")

    # ── optimization ────────────────────────────────────────────────────────

    def optimize(
        self,
        keyframe_poses: dict[int, np.ndarray],
        *,
        origin_id:      int,
        max_iters:      int = 30,
    ) -> tuple[dict[int, np.ndarray], dict]:
        """Run Levenberg-Marquardt over the pose graph.

        Returns ``(new_poses, info)`` where ``new_poses[kf_id]`` is the
        optimized 4x4 SE(3) for every keyframe present in the input, and
        ``info`` has timing + error statistics.

        Edges referencing keyframe ids not in ``keyframe_poses`` are silently
        skipped (lets us call this without a separate "edges-only-touching-
        existing-kfs" filter after eviction).
        """
        from loguru import logger
        import gtsam

        if not self._edges:
            return dict(keyframe_poses), {"skipped": "no edges"}

        t0 = time.monotonic()
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        for kf_id, pose in keyframe_poses.items():
            initial.insert(int(kf_id), gtsam.Pose3(np.asarray(pose, dtype=np.float64)))

        # Anchor the origin keyframe so the system is well-determined.  Hard
        # prior — we never want the origin to move.
        prior_noise = gtsam.noiseModel.Constrained.All(6)
        graph.add(gtsam.PriorFactorPose3(
            int(origin_id),
            gtsam.Pose3(np.asarray(keyframe_poses[origin_id], dtype=np.float64)),
            prior_noise,
        ))

        used = 0
        for e in self._edges:
            if e.src_id not in keyframe_poses or e.dst_id not in keyframe_poses:
                continue
            used += 1
            # Inlier count → information.  Heuristic: sigma_translation ~ 1 cm
            # at 200 inliers, 5 cm at 20 inliers; sigma_rotation ~ 0.5 deg at
            # 200 inliers, 2.5 deg at 20 inliers.  Cap at a sane floor so a
            # huge inlier count doesn't make any single edge unmovable.
            n = max(e.inliers, 1)
            sigma_t = max(0.005, 2.0 / n)               # metres
            sigma_r = max(np.deg2rad(0.25), 0.1 / n)    # radians
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
                sigma_r, sigma_r, sigma_r,
                sigma_t, sigma_t, sigma_t,
            ]))
            graph.add(gtsam.BetweenFactorPose3(
                int(e.src_id), int(e.dst_id),
                gtsam.Pose3(e.rel_pose), noise,
            ))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(max_iters))
        try:
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        except Exception as exc:                                # noqa: BLE001
            logger.opt(exception=True).warning("PGO: optimizer raised — keeping pre-opt poses ({})", exc)
            return dict(keyframe_poses), {"error": str(exc)}

        new_poses: dict[int, np.ndarray] = {}
        for kf_id in keyframe_poses:
            new_poses[int(kf_id)] = np.asarray(
                result.atPose3(int(kf_id)).matrix(), dtype=np.float64,
            )

        info = {
            "edges":       used,
            "loops":       self.num_loops(),
            "elapsed_ms":  (time.monotonic() - t0) * 1000.0,
            "initial_err": float(graph.error(initial)),
            "final_err":   float(graph.error(result)),
        }
        logger.info(
            "PGO  edges={}  loops={}  err {:.3g}→{:.3g}  ({:.0f} ms)",
            info["edges"], info["loops"],
            info["initial_err"], info["final_err"], info["elapsed_ms"],
        )
        return new_poses, info
