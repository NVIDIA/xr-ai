# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TUM RGB-D benchmark for pose-mcp.

Runs the in-process ``Localizer`` over a TUM RGB-D sequence (default:
``freiburg1_xyz``), times each stage, and reports the Absolute Trajectory
Error (ATE-RMSE) of the estimated trajectory after Sim(3) alignment to
the ground truth — monocular SLAM has a free scale + global pose, so
Sim(3) (rotation + translation + scale) is the right alignment.

Dataset is expected at ``/tmp/slam-bench/rgbd_dataset_<seq>/``.  Pass
``--rgbd /path/to/dir`` to point at a different location.

Two backends:

* ``--depth-backend moge`` — runs MoGe to get per-keyframe metric depth
  exactly the way pose-mcp does in production.  GPU recommended.
* ``--depth-backend kinect`` — substitutes the Kinect depth map TUM
  ships with each frame for MoGe's output.  Bypasses the heaviest
  model so we can isolate the matching + PnP cost.

Two intrinsics modes:

* ``--intrinsics tum`` — fixed TUM freiburg1 K (fx=517.3 / fy=516.5
  / cx=318.6 / cy=255.3).  No FOV calibration, no MoGe FOV trust.
* ``--intrinsics moge`` — let pose-mcp calibrate FOV via MoGe as
  normal (will fall back to MoGe-derived per-frame intrinsics).

Usage::

    cd agent-mcp-servers/pose-mcp
    uv run python bench/tum_bench.py --max-frames 100 --intrinsics tum --depth-backend kinect

The script writes a CSV next to the dataset with per-frame timings +
estimated poses, prints an ATE summary at the end.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import sys
import time
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

# Make the pose-mcp source tree importable without a package install.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from pose_mcp_server.backends   import FrameFeatures, GeometryFrame   # noqa: E402
from pose_mcp_server.localizer  import Localizer                       # noqa: E402
from pose_mcp_server.store      import KeyframeStore                   # noqa: E402


# ── dataset readers ──────────────────────────────────────────────────────

@dataclasses.dataclass
class Frame:
    ts: float
    rgb_path: pathlib.Path
    depth_path: pathlib.Path | None


def _read_assoc(rgb_txt: pathlib.Path, depth_txt: pathlib.Path | None,
                max_offset_s: float = 0.02) -> list[Frame]:
    """Read TUM RGB + (optionally) depth list files and associate each
    RGB timestamp with the closest depth frame within ``max_offset_s``."""
    def _rows(path: pathlib.Path):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            yield float(parts[0]), pathlib.Path(parts[1])
    base = rgb_txt.parent
    rgb_rows   = list(_rows(rgb_txt))
    depth_rows = list(_rows(depth_txt)) if depth_txt and depth_txt.exists() else []
    out: list[Frame] = []
    for ts, fname in rgb_rows:
        dpath = None
        if depth_rows:
            # Binary-search nearest depth ts.
            ds = [abs(ts - dt) for dt, _ in depth_rows]
            j  = int(np.argmin(ds))
            if ds[j] <= max_offset_s:
                dpath = base / depth_rows[j][1]
        out.append(Frame(ts=ts, rgb_path=base / fname, depth_path=dpath))
    return out


def _read_gt(path: pathlib.Path) -> list[tuple[float, np.ndarray]]:
    """Return ``[(timestamp, T_world_camera 4x4)]``.  TUM's quaternion
    ordering is ``(qx, qy, qz, qw)`` — we keep our internal ``[w,x,y,z]``
    convention consistent and just convert here."""
    out: list[tuple[float, np.ndarray]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts = float(parts[0])
        t  = np.array(parts[1:4], dtype=np.float64)
        qx, qy, qz, qw = [float(x) for x in parts[4:8]]
        # Quaternion → rotation matrix (right-handed, x-y-z order).
        n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        R = np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        out.append((ts, T))
    return out


def _align_gt_to_frames(gt: list[tuple[float, np.ndarray]],
                       frame_ts: list[float],
                       max_offset_s: float = 0.05) -> list[np.ndarray | None]:
    """Pick the ground-truth pose nearest each frame timestamp.  Returns
    ``None`` for frames that have no nearby GT."""
    gt_ts = np.array([t for t, _ in gt])
    aligned: list[np.ndarray | None] = []
    for ft in frame_ts:
        idx = int(np.argmin(np.abs(gt_ts - ft)))
        if abs(gt_ts[idx] - ft) <= max_offset_s:
            aligned.append(gt[idx][1])
        else:
            aligned.append(None)
    return aligned


# ── synthetic backends (skip MoGe / XFeat heavyweight loads) ─────────────

class _KinectGeometry:
    """Substitutes the Kinect depth map for MoGe's output.

    Builds a metric ``(H, W, 3)`` point cloud directly from the depth
    image and the TUM intrinsics, lets the Localizer treat it as a
    perfect MoGe pass.  ``is_calibrated`` always True (we have known K),
    ``pinned_fov_deg`` returns the fov derived from K.
    """
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 depth_lookup: dict[str, pathlib.Path],
                 native_size: tuple[int, int] = (640, 480)) -> None:
        # Native = the resolution the TUM intrinsics were calibrated at.
        # The Localizer's `image_max_edge` may have shrunk the actual
        # input image before reaching us — we scale K to match below.
        self._fx_native = fx; self._fy_native = fy
        self._cx_native = cx; self._cy_native = cy
        self._native_W, self._native_H = native_size
        self._depth_lookup = depth_lookup       # rgb-stem → depth path
        self._current_key: str | None = None
        # TUM depth is in millimetres scaled by 5000 (16-bit PNG).
        self._depth_scale = 1.0 / 5000.0

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def pinned_fov_deg(self) -> float:
        # FOV is invariant under uniform image scaling, so the native fx +
        # native W give the right answer regardless of how the input has
        # been resized.
        return float(2.0 * np.degrees(np.arctan(0.5 * self._native_W / self._fx_native)))

    def __call__(self, image_rgb: np.ndarray) -> GeometryFrame:
        H, W = image_rgb.shape[:2]
        # Scale K to the actual input resolution (uniform scale assumed).
        sx = W / float(self._native_W)
        sy = H / float(self._native_H)
        fx = self._fx_native * sx
        fy = self._fy_native * sy
        cx = self._cx_native * sx
        cy = self._cy_native * sy

        dpath = self._depth_lookup.get(self._current_key)
        if dpath is None or not dpath.exists():
            return GeometryFrame(
                points3d=np.zeros((H, W, 3), dtype=np.float32),
                mask=np.zeros((H, W), dtype=bool),
                fov_deg=self.pinned_fov_deg, width=W, height=H,
            )
        depth_mm = np.asarray(Image.open(dpath)).astype(np.float32)
        depth_m  = depth_mm * self._depth_scale
        if depth_m.shape != (H, W):
            depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        Z = depth_m
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy
        pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        # Drop invalid depth (== 0) plus a small border like MoGe path.
        b = 10
        mask = (Z > 0.05) & (Z < 10.0)
        mask[:b, :] = False; mask[-b:, :] = False
        mask[:, :b] = False; mask[:, -b:] = False
        return GeometryFrame(points3d=pts, mask=mask,
                             fov_deg=self.pinned_fov_deg, width=W, height=H)


# ── ATE / trajectory alignment ───────────────────────────────────────────

def _sim3_align(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """Umeyama Sim(3) alignment of an estimated trajectory to ground
    truth.  Both inputs are ``(N, 3)`` translation arrays.  Returns
    ``(transformed_est, scale)``.

    Monocular SLAM has a free 7-DoF gauge: rotation, translation, scale.
    Without this the raw ATE is meaningless.
    """
    N = len(est)
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)
    Ec = est - mu_e
    Gc = gt  - mu_g
    # Σ_xy = (1/N) Ec^T Gc.  We compute it directly so the scale formula's
    # normalisation is unambiguous (a previous version that worked off
    # `Ec.T @ Gc` was off by exactly factor N — the trace was N×Σ_xy's).
    Sigma_xy = (Ec.T @ Gc) / float(N)
    U, S, Vt = np.linalg.svd(Sigma_xy)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    # Umeyama: s = trace(S·D) / σ²_x where σ²_x = mean ||x_i - μ_x||².
    var_e = (Ec ** 2).sum() / float(N)
    scale = float((S * np.diag(D)).sum() / var_e) if var_e > 1e-12 else 1.0
    t = mu_g - scale * R @ mu_e
    aligned = (scale * (R @ est.T)).T + t
    return aligned, scale


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rgbd", type=pathlib.Path,
                   default=pathlib.Path("/tmp/slam-bench/rgbd_dataset_freiburg1_xyz"))
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--depth-backend", choices=["moge", "kinect"],
                   default="kinect",
                   help="MoGe runs the real model; kinect uses TUM's depth maps")
    p.add_argument("--intrinsics", choices=["tum", "moge"], default="tum")
    p.add_argument("--device", default="auto")
    p.add_argument("--image-max-edge", type=int, default=384)
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--lighterglue-min-conf", type=float, default=0.05)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("/tmp/slam-bench/bench_out.csv"))
    p.add_argument("--profile", action="store_true",
                   help="Wrap each backend call so the per-stage cost (geometry/extract/match/pnp) is printed at the end.")
    args = p.parse_args()

    rgb_txt   = args.rgbd / "rgb.txt"
    depth_txt = args.rgbd / "depth.txt"
    gt_txt    = args.rgbd / "groundtruth.txt"

    print(f"[bench] dataset: {args.rgbd}")
    frames = _read_assoc(rgb_txt, depth_txt)
    if args.max_frames:
        frames = frames[: args.max_frames]
    gt = _read_gt(gt_txt)
    gt_poses = _align_gt_to_frames(gt, [f.ts for f in frames])
    print(f"[bench] frames: {len(frames)}  gt rows: {len(gt)}  "
          f"gt-matched: {sum(g is not None for g in gt_poses)}")

    # Backends.
    if args.depth_backend == "moge":
        from pose_mcp_server.backends import MoGeBackend
        geom_backend = MoGeBackend(
            model_name="Ruicheng/moge-2-vits-normal",
            device=args.device,
            fov_x_deg=None if args.intrinsics == "moge" else 60.0,
            calibration_frames=0 if args.intrinsics == "tum" else 8,
        )
    else:
        depth_lookup = {f.rgb_path.stem: f.depth_path for f in frames if f.depth_path}
        geom_backend = _KinectGeometry(
            fx=517.3, fy=516.5, cx=318.6, cy=255.3,
            depth_lookup=depth_lookup,
        )

    from pose_mcp_server.backends import XFeatBackend
    feat_backend = XFeatBackend(
        device=args.device, top_k=args.top_k,
        min_conf=args.lighterglue_min_conf,
    )

    # Optional per-stage profiler.  Wraps the geometry + feature + match
    # callables and accumulates wall-clock per call.  Cheap to leave on.
    stage_times: dict[str, list[float]] = {
        "geometry": [], "extract": [], "match": [], "process": [],
    }
    if args.profile:
        _orig_geom_call = geom_backend.__call__
        def _geom_timed(img):
            t0 = time.perf_counter()
            r = _orig_geom_call(img)
            stage_times["geometry"].append((time.perf_counter() - t0) * 1000)
            return r
        # MoGeBackend / KinectGeometry are dataclass-ish — just rebind.
        geom_backend.__call__ = _geom_timed                              # type: ignore[assignment]

        _orig_extract = feat_backend.extract
        def _extract_timed(img):
            t0 = time.perf_counter()
            r = _orig_extract(img)
            stage_times["extract"].append((time.perf_counter() - t0) * 1000)
            return r
        feat_backend.extract = _extract_timed                            # type: ignore[assignment]

        _orig_match = feat_backend.match
        def _match_timed(a, b):
            t0 = time.perf_counter()
            r = _orig_match(a, b)
            stage_times["match"].append((time.perf_counter() - t0) * 1000)
            return r
        feat_backend.match = _match_timed                                # type: ignore[assignment]

    import tempfile
    with tempfile.TemporaryDirectory(prefix="pose_bench_map_") as map_dir:
        store = KeyframeStore(pathlib.Path(map_dir))
        loc   = Localizer(
            store=store, geometry=geom_backend, features=feat_backend,
            min_inliers=args.min_inliers,
            image_max_edge=args.image_max_edge,
            pose_graph=None,                # PGO off for timing-clean bench
        )

        # ── run loop ───────────────────────────────────────────────────
        rows: list[dict] = []
        est_traj: list[tuple[int, np.ndarray]] = []
        for i, f in enumerate(frames):
            with Image.open(f.rgb_path) as img:
                rgb = np.asarray(img.convert("RGB"))
            # Hand the kinect-depth lookup the rgb stem the localizer is
            # about to consume.
            if isinstance(geom_backend, _KinectGeometry):
                geom_backend._current_key = f.rgb_path.stem

            t0 = time.perf_counter()
            r = loc.process(rgb, ts_us=int(f.ts * 1_000_000))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            stage_times["process"].append(dt_ms)

            row = {
                "i": i, "ts": f.ts, "state": r.state,
                "inliers": r.num_inliers, "kfs": r.num_keyframes,
                "elapsed_ms": dt_ms,
            }
            if r.pose is not None:
                t = r.pose[:3, 3]
                row.update({"tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2])})
                est_traj.append((i, t.astype(np.float64)))
            rows.append(row)

            if i % 25 == 0:
                print(f"[bench] frame {i:4d}/{len(frames)}  state={r.state:>11s}  "
                      f"inliers={r.num_inliers:>4d}  kfs={r.num_keyframes:>3d}  "
                      f"elapsed={dt_ms:6.1f} ms")

    # ── write CSV ──────────────────────────────────────────────────────
    fieldnames = ["i", "ts", "state", "inliers", "kfs", "elapsed_ms", "tx", "ty", "tz"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[bench] wrote {args.out}")

    # ── summary ────────────────────────────────────────────────────────
    elapsed = np.array([r["elapsed_ms"] for r in rows])
    print()
    print("== latency ==")
    print(f"  mean: {elapsed.mean():.1f} ms   ({1000/elapsed.mean():.1f} FPS)")
    print(f"  p50 : {np.percentile(elapsed, 50):.1f} ms")
    print(f"  p95 : {np.percentile(elapsed, 95):.1f} ms")
    print(f"  max : {elapsed.max():.1f} ms")
    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    print(f"  states: {states}")

    if args.profile:
        print()
        print("== per-stage (warm; first frame excluded) ==")
        for k, times in stage_times.items():
            arr = np.array(times[1:]) if len(times) > 1 else np.array(times)
            if not arr.size:
                continue
            print(f"  {k:<10s} mean {arr.mean():6.1f} ms  p50 {np.median(arr):6.1f}  "
                  f"p95 {np.percentile(arr, 95):6.1f}  n={len(arr)}")

    # ATE — Sim(3)-align estimated translations to gt.
    if est_traj and any(g is not None for g in gt_poses):
        idxs   = [i for i, _ in est_traj]
        est_xy = np.stack([t for _, t in est_traj], axis=0)
        gt_xy: list[np.ndarray] = []
        keep:  list[int] = []
        for k, i in enumerate(idxs):
            if gt_poses[i] is not None:
                gt_xy.append(gt_poses[i][:3, 3])
                keep.append(k)
        if len(keep) >= 4:
            est_xy = est_xy[keep]
            gt_xy  = np.stack(gt_xy, axis=0)
            aligned, scale = _sim3_align(est_xy, gt_xy)
            errs = np.linalg.norm(aligned - gt_xy, axis=1)
            print()
            print(f"== ATE (Sim(3)-aligned, N={len(keep)}) ==")
            print(f"  scale recovered: {scale:.3f}")
            print(f"  RMSE: {np.sqrt((errs**2).mean()) * 100:.2f} cm")
            print(f"  mean: {errs.mean() * 100:.2f} cm")
            print(f"  max : {errs.max() * 100:.2f} cm")
        else:
            print("  (too few localized frames to compute ATE)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
