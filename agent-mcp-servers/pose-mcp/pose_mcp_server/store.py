# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent keyframe database.

Layout on disk::

    <map_dir>/
      manifest.json                # version + intrinsics convention
      keyframes.jsonl              # one row per keyframe: id, ts_us, pose_4x4, fov_deg
      kf{id}/keypoints.npy         # (N, 2) float32
      kf{id}/descriptors.npy       # (N, D) float32
      kf{id}/points3d.npy          # (H, W, 3) float16 metric points in keyframe frame
      kf{id}/mask.npy              # (H, W)   bool      where points3d is valid
      kf{id}/image.png             # (H, W, 3) uint8 RGB — optional, used for
                                   #   colouring the point cloud in viz sinks

JSONL append + ``os.replace`` atomicity gives us a crash-safe map.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import tempfile
import time

import numpy as np
from PIL import Image as PILImage

_MANIFEST_VERSION = 1


@dataclasses.dataclass(frozen=True)
class Keyframe:
    id:        int
    ts_us:     int
    pose:      np.ndarray   # 4x4 SE(3), world ← keyframe
    fov_deg:   float        # horizontal FOV from MoGe
    kp:        np.ndarray   # (N, 2) float32
    desc:      np.ndarray   # (N, D) float32
    pts3d:     np.ndarray   # (H, W, 3) float16
    mask:      np.ndarray   # (H, W) bool
    # RGB image the keyframe was built from.  Optional so legacy maps
    # written before this field existed still load — viz sinks fall back
    # to uncoloured point clouds when None.
    image_rgb: np.ndarray | None = None   # (H, W, 3) uint8


class KeyframeStore:
    """File-backed list of keyframes.  All mutations are atomic — a crash
    mid-write leaves the previous state intact."""

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root
        self._keyframes: list[Keyframe] = []
        self._next_id  = 0
        self._root.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest()
        self._load()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _ensure_manifest(self) -> None:
        path = self._root / "manifest.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("version") != _MANIFEST_VERSION:
                raise RuntimeError(
                    f"pose-map manifest version mismatch at {path} "
                    f"(found {data.get('version')!r}, expected {_MANIFEST_VERSION}). "
                    "Delete the map directory or migrate manually."
                )
            return
        self._atomic_write(path, json.dumps({
            "version":     _MANIFEST_VERSION,
            "created_us":  int(time.time() * 1_000_000),
            "convention": {
                "pose":         "T_world_keyframe, 4x4 row-major, world coords in metres",
                "quaternion":   "[w, x, y, z]",
                "points3d":     "(H, W, 3) metric in keyframe frame, float16; invalid → NaN",
            },
        }, indent=2).encode())

    def _load(self) -> None:
        jl = self._root / "keyframes.jsonl"
        if not jl.exists():
            return
        for line in jl.read_text().splitlines():
            row = json.loads(line)
            kf_dir = self._kf_dir(row["id"])
            img_path = kf_dir / "image.png"
            image_rgb: np.ndarray | None = None
            if img_path.exists():
                with PILImage.open(img_path) as img:
                    image_rgb = np.asarray(img.convert("RGB"))
            kf = Keyframe(
                id      = int(row["id"]),
                ts_us   = int(row["ts_us"]),
                pose    = np.asarray(row["pose"], dtype=np.float64).reshape(4, 4),
                fov_deg = float(row["fov_deg"]),
                kp      = np.load(kf_dir / "keypoints.npy"),
                desc    = np.load(kf_dir / "descriptors.npy"),
                pts3d   = np.load(kf_dir / "points3d.npy"),
                mask    = np.load(kf_dir / "mask.npy"),
                image_rgb = image_rgb,
            )
            self._keyframes.append(kf)
            self._next_id = max(self._next_id, kf.id + 1)

    # ── queries ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._keyframes)

    def all(self) -> list[Keyframe]:
        return list(self._keyframes)

    def last(self) -> Keyframe | None:
        return self._keyframes[-1] if self._keyframes else None

    def stats(self) -> dict:
        if not self._keyframes:
            return {
                "num_keyframes": 0,
                "map_dir":       str(self._root),
                "origin_set":    False,
            }
        first = self._keyframes[0]
        last  = self._keyframes[-1]
        return {
            "num_keyframes":   len(self._keyframes),
            "map_dir":         str(self._root),
            "origin_set":      True,
            "earliest_ts_us":  int(first.ts_us),
            "latest_ts_us":    int(last.ts_us),
            "last_pose":       last.pose.tolist(),
            "last_fov_deg":    float(last.fov_deg),
        }

    # ── mutations ───────────────────────────────────────────────────────────

    def append(
        self, *,
        ts_us:     int,
        pose:      np.ndarray,
        fov_deg:   float,
        kp:        np.ndarray,
        desc:      np.ndarray,
        pts3d:     np.ndarray,
        mask:      np.ndarray,
        image_rgb: np.ndarray | None = None,
    ) -> Keyframe:
        kf = Keyframe(
            id=self._next_id, ts_us=int(ts_us),
            pose=np.asarray(pose, dtype=np.float64).reshape(4, 4),
            fov_deg=float(fov_deg),
            kp=np.ascontiguousarray(kp,  dtype=np.float32),
            desc=np.ascontiguousarray(desc, dtype=np.float32),
            pts3d=np.ascontiguousarray(pts3d, dtype=np.float16),
            mask=np.ascontiguousarray(mask, dtype=bool),
            image_rgb=(
                np.ascontiguousarray(image_rgb, dtype=np.uint8)
                if image_rgb is not None else None
            ),
        )
        kf_dir = self._kf_dir(kf.id)
        kf_dir.mkdir(parents=True, exist_ok=True)
        np.save(kf_dir / "keypoints.npy",   kf.kp)
        np.save(kf_dir / "descriptors.npy", kf.desc)
        np.save(kf_dir / "points3d.npy",    kf.pts3d)
        np.save(kf_dir / "mask.npy",        kf.mask)
        if kf.image_rgb is not None:
            # PNG keeps the file small enough that 200-keyframe maps don't
            # explode on disk (~300 KB / kf at 640x480 vs. ~1 MB raw .npy).
            PILImage.fromarray(kf.image_rgb, "RGB").save(
                kf_dir / "image.png", "PNG", optimize=True,
            )
        # Append-then-flush: a crash between np.save() above and the JSONL
        # append leaves orphan kf{id} dirs but no half-recorded keyframes.
        # Orphans are harmless and easy to spot when cleaning the map dir.
        row = {
            "id":      kf.id,
            "ts_us":   kf.ts_us,
            "pose":    kf.pose.tolist(),
            "fov_deg": kf.fov_deg,
        }
        with (self._root / "keyframes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._keyframes.append(kf)
        self._next_id += 1
        return kf

    def evict_oldest(self) -> None:
        if not self._keyframes:
            return
        victim = self._keyframes.pop(0)
        kf_dir = self._kf_dir(victim.id)
        for p in kf_dir.glob("*"):
            p.unlink()
        kf_dir.rmdir()
        # Rewrite JSONL without the evicted row.  Atomic via tmp+rename.
        tmp = self._root / "keyframes.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for kf in self._keyframes:
                f.write(json.dumps({
                    "id": kf.id, "ts_us": kf.ts_us,
                    "pose": kf.pose.tolist(), "fov_deg": kf.fov_deg,
                }) + "\n")
        os.replace(tmp, self._root / "keyframes.jsonl")

    def reset(self) -> None:
        self._keyframes.clear()
        self._next_id = 0
        for p in sorted(self._root.glob("kf*"), reverse=True):
            if p.is_dir():
                for f in p.glob("*"):
                    f.unlink()
                p.rmdir()
        jl = self._root / "keyframes.jsonl"
        if jl.exists():
            jl.unlink()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _kf_dir(self, idx: int) -> pathlib.Path:
        return self._root / f"kf{idx:06d}"

    @staticmethod
    def _atomic_write(path: pathlib.Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise
