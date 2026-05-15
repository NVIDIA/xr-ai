# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pose-mcp.

These cover the dependency-light pieces:

* Quaternion ↔ rotation-matrix round trips and SE(3) helpers.
* ``KeyframeStore`` persistence: append → reload from disk preserves rows,
  ``reset_map`` wipes both metadata and per-keyframe arrays, eviction trims
  the oldest entry and rewrites the JSONL atomically.
* The ``Localizer`` state machine via two synthetic fake backends — proving
  the empty/bootstrap/localized transitions and that pose composition is
  consistent with the keyframe pose chain.

The MoGe and XFeat backends themselves require GPU + multi-GB model weights
to exercise meaningfully and are covered by a ``@pytest.mark.gpu`` test in
``test_gpu_pose_mcp.py``.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pose_mcp_server.backends  import FrameFeatures, GeometryFrame
from pose_mcp_server.geometry  import (compose_se3, invert_se3, make_se3,
                                       quat_to_rmat, rmat_to_quat,
                                       se3_rotation_deg, se3_translation)
from pose_mcp_server.localizer import Localizer, PoseResult
from pose_mcp_server.pgo       import PoseEdge, PoseGraph
from pose_mcp_server.store     import Keyframe, KeyframeStore


# ── geometry helpers ──────────────────────────────────────────────────────────

def _axis_angle_R(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@pytest.mark.parametrize("axis,angle", [
    (np.array([1.0, 0.0, 0.0]), 0.0),
    (np.array([1.0, 0.0, 0.0]), 0.5),
    (np.array([0.0, 1.0, 0.0]), 1.2),
    (np.array([0.3, 0.7, 0.4]), -0.9),
])
def test_quat_rmat_roundtrip(axis, angle):
    R = _axis_angle_R(axis, angle)
    q = rmat_to_quat(R)
    R2 = quat_to_rmat(q)
    np.testing.assert_allclose(R2, R, atol=1e-10)


def test_invert_se3_roundtrip():
    R = _axis_angle_R(np.array([0.1, 0.9, 0.2]), 0.8)
    T = make_se3(R, np.array([1.5, -0.4, 0.3]))
    Tinv = invert_se3(T)
    np.testing.assert_allclose(compose_se3(T, Tinv), np.eye(4), atol=1e-10)
    np.testing.assert_allclose(compose_se3(Tinv, T), np.eye(4), atol=1e-10)


def test_se3_translation_and_rotation_deg():
    R = _axis_angle_R(np.array([0.0, 1.0, 0.0]), np.deg2rad(30.0))
    T = make_se3(R, np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(se3_translation(T), [2.0, 0.0, 0.0])
    assert abs(se3_rotation_deg(np.eye(4), T) - 30.0) < 1e-6


# ── persistence ───────────────────────────────────────────────────────────────

def _fake_kf_args(seed: int, H: int = 4, W: int = 5):
    rng = np.random.default_rng(seed)
    kp    = rng.random((10, 2), dtype=np.float32) * 4.0
    desc  = rng.random((10, 8), dtype=np.float32)
    pts3d = rng.random((H, W, 3), dtype=np.float32).astype(np.float16)
    mask  = np.ones((H, W), dtype=bool)
    pose  = make_se3(np.eye(3), [seed * 1.0, 0.0, 0.0])
    return dict(ts_us=seed * 1000, pose=pose, fov_deg=90.0,
                kp=kp, desc=desc, pts3d=pts3d, mask=mask)


def test_store_append_reload(tmp_path: pathlib.Path):
    store = KeyframeStore(tmp_path)
    store.append(**_fake_kf_args(1))
    store.append(**_fake_kf_args(2))
    assert len(store) == 2

    reloaded = KeyframeStore(tmp_path)
    assert len(reloaded) == 2
    np.testing.assert_allclose(reloaded.all()[0].pose, store.all()[0].pose)
    np.testing.assert_allclose(reloaded.all()[1].kp,   store.all()[1].kp)
    np.testing.assert_allclose(reloaded.all()[0].pts3d.astype(np.float32),
                               store.all()[0].pts3d.astype(np.float32))


def test_store_evict_oldest(tmp_path: pathlib.Path):
    store = KeyframeStore(tmp_path)
    for i in range(3):
        store.append(**_fake_kf_args(i + 1))
    first_id = store.all()[0].id
    store.evict_oldest()
    assert len(store) == 2
    assert store.all()[0].id != first_id
    # On-disk reload sees the same view.
    reloaded = KeyframeStore(tmp_path)
    assert len(reloaded) == 2
    assert reloaded.all()[0].id != first_id


def test_store_reset(tmp_path: pathlib.Path):
    store = KeyframeStore(tmp_path)
    store.append(**_fake_kf_args(1))
    store.reset()
    assert len(store) == 0
    # Per-keyframe dirs should be gone too.
    assert not list(tmp_path.glob("kf*"))
    # Fresh load sees an empty store.
    assert len(KeyframeStore(tmp_path)) == 0


# ── localizer state machine ──────────────────────────────────────────────────

class _FakeGeometry:
    """Returns a 1m-deep planar point cloud on a known grid."""
    def __init__(self, W: int = 64, H: int = 48, fov_deg: float = 60.0):
        self.W, self.H, self.fov_deg = W, H, fov_deg
        fx = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
        self._K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)
        self._depth = 1.0

    def __call__(self, image_rgb: np.ndarray) -> GeometryFrame:
        xs, ys = np.meshgrid(np.arange(self.W), np.arange(self.H))
        Z = np.full((self.H, self.W), self._depth, dtype=np.float32)
        X = (xs - self._K[0, 2]) * Z / self._K[0, 0]
        Y = (ys - self._K[1, 2]) * Z / self._K[1, 1]
        pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        mask = np.ones((self.H, self.W), dtype=bool)
        return GeometryFrame(points3d=pts, mask=mask, fov_deg=self.fov_deg,
                             width=self.W, height=self.H)


class _FakeFeatures:
    """Deterministic, identity-matching feature backend.

    Each frame's "features" are a fixed grid of keypoints with descriptors that
    encode the keypoint index — so ``match(a, b)`` is just an index-equality
    test.  That gives the localizer the perfect 2D-2D correspondences a real
    XFeat+LighterGlue pass would (ideally) produce, isolating the PnP path
    from feature-matching quality.
    """
    def __init__(self, W: int = 64, H: int = 48, n: int = 100):
        rng = np.random.default_rng(0)
        xs = rng.uniform(4, W - 4, size=n).astype(np.float32)
        ys = rng.uniform(4, H - 4, size=n).astype(np.float32)
        self._kp = np.stack([xs, ys], axis=1)
        self._desc = np.eye(n, dtype=np.float32)  # row i ≡ index i

    def extract(self, image_rgb: np.ndarray) -> FrameFeatures:
        return FrameFeatures(kp=self._kp.copy(), desc=self._desc.copy())

    def match(self, a: FrameFeatures, b: FrameFeatures) -> np.ndarray:
        n = min(len(a.kp), len(b.kp))
        return np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.int32)


def _white_image(W: int, H: int) -> np.ndarray:
    return np.full((H, W, 3), 255, dtype=np.uint8)


def test_localizer_empty_then_localized(tmp_path: pathlib.Path):
    store = KeyframeStore(tmp_path)
    geom  = _FakeGeometry()
    feats = _FakeFeatures(W=geom.W, H=geom.H)
    loc   = Localizer(store=store, geometry=geom, features=feats, min_inliers=10)

    img = _white_image(geom.W, geom.H)

    r1 = loc.process(img, ts_us=1_000_000)
    assert r1.state == "empty"
    assert r1.num_keyframes == 1
    assert r1.translation == [0.0, 0.0, 0.0]

    r2 = loc.process(img, ts_us=2_000_000)
    assert r2.state == "localized"
    assert r2.num_inliers >= 10
    # PnP-RANSAC introduces sub-cm noise even on perfect synthetic input —
    # the contract here is "we recovered the identity pose", not bit-exact.
    np.testing.assert_allclose(r2.translation, [0.0, 0.0, 0.0], atol=5e-3)
    np.testing.assert_allclose(r2.quaternion,  [1.0, 0.0, 0.0, 0.0], atol=5e-3)


class _PlanarScene:
    """Self-consistent geometry + feature fake.

    Places N landmarks on a fronto-parallel plane at ``Z=plane_depth`` in the
    keyframe's camera frame.  Acts as both backends:

    * ``geometry()`` returns a point map of the same plane, so when the
      localizer lifts a keyframe pixel into 3D the answer matches the true
      landmark coordinate (no model-vs-reality mismatch corrupting PnP).
    * ``extract()`` projects the landmarks through ``self.pose`` (the
      world-frame pose of the *current* camera) and returns the resulting
      pixel positions.  Mutating ``pose`` between calls is how the test
      simulates camera motion.

    Identity-encoded one-hot descriptors make ``match()`` argmax-trivial,
    isolating the test from feature-matching quality.
    """
    def __init__(self, *, W: int = 128, H: int = 96, fov_deg: float = 70.0,
                 n_landmarks: int = 80, plane_depth: float = 2.0, seed: int = 7):
        self.W, self.H, self.fov_deg = W, H, fov_deg
        fx = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
        self._fx, self._cx, self._cy = fx, W / 2.0, H / 2.0
        self._plane_depth = plane_depth
        # Landmarks live on Z=plane_depth in keyframe coords; bound (X,Y) so
        # every landmark stays on-image for the small motions we test.
        rng = np.random.default_rng(seed)
        margin = 0.6   # fraction of FOV reserved for camera motion
        X = rng.uniform(-margin, margin, size=n_landmarks) * plane_depth
        Y = rng.uniform(-margin, margin, size=n_landmarks) * plane_depth * (H / W)
        Z = np.full(n_landmarks, plane_depth)
        self._landmarks_kf = np.stack([X, Y, Z], axis=1).astype(np.float64)
        self._desc = np.eye(n_landmarks, dtype=np.float32)
        self.pose = np.eye(4, dtype=np.float64)   # world ← camera

    def geometry(self, image_rgb: np.ndarray) -> GeometryFrame:
        xs, ys = np.meshgrid(np.arange(self.W), np.arange(self.H))
        Z = np.full((self.H, self.W), self._plane_depth, dtype=np.float32)
        X = (xs - self._cx) * Z / self._fx
        Y = (ys - self._cy) * Z / self._fx
        pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        mask = np.ones((self.H, self.W), dtype=bool)
        return GeometryFrame(points3d=pts, mask=mask, fov_deg=self.fov_deg,
                             width=self.W, height=self.H)

    def extract(self, image_rgb: np.ndarray) -> FrameFeatures:
        # T_cam_world projects world-frame points (== keyframe coords here
        # because keyframe 0 is at world origin) into the current camera.
        T_cam_world = np.linalg.inv(self.pose)
        P_h  = np.hstack([self._landmarks_kf, np.ones((len(self._landmarks_kf), 1))])
        P_c  = (T_cam_world @ P_h.T).T[:, :3]
        in_front = P_c[:, 2] > 0.05
        u = self._fx * P_c[:, 0] / np.where(in_front, P_c[:, 2], 1.0) + self._cx
        v = self._fx * P_c[:, 1] / np.where(in_front, P_c[:, 2], 1.0) + self._cy
        ok = in_front & (u >= 1) & (u < self.W - 1) & (v >= 1) & (v < self.H - 1)
        return FrameFeatures(
            kp=np.stack([u, v], axis=1)[ok].astype(np.float32),
            desc=self._desc[ok],
        )

    @staticmethod
    def match(a: FrameFeatures, b: FrameFeatures) -> np.ndarray:
        # Each row of desc is a one-hot landmark id; pair by argmax-of-similarity.
        sim = a.desc @ b.desc.T
        if sim.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        j = sim.argmax(axis=1)
        keep = sim[np.arange(len(j)), j] > 0.5
        return np.stack([np.arange(len(j))[keep], j[keep]], axis=1).astype(np.int32)


def test_localizer_recovers_known_pose(tmp_path: pathlib.Path):
    """A virtual camera translated 0.3 m right + 0.1 m forward and yawed 15°
    must come back as the recovered ``T_world_cam`` — within PnP noise."""
    scene = _PlanarScene()
    store = KeyframeStore(tmp_path)
    loc   = Localizer(store=store, geometry=scene.geometry, features=scene,
                      min_inliers=10, min_translation_m=999, min_rotation_deg=999)

    img = _white_image(scene.W, scene.H)

    # First frame at origin → empty.
    r1 = loc.process(img, ts_us=1)
    assert r1.state == "empty"

    # Move the virtual camera: 0.3 m +X, 0.1 m +Z, 15° yaw about Y.
    yaw = np.deg2rad(15.0)
    R   = np.array([
        [ np.cos(yaw), 0.0, np.sin(yaw)],
        [ 0.0,         1.0, 0.0       ],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    t = np.array([0.30, 0.0, 0.10])
    scene.pose = make_se3(R, t)

    r2 = loc.process(img, ts_us=2)
    assert r2.state == "localized"
    assert r2.num_inliers >= 10

    np.testing.assert_allclose(r2.translation, t, atol=5e-2)
    # Rotation: compare via the angle between recovered R and the truth.
    R_recovered = quat_to_rmat(np.array(r2.quaternion))
    delta_deg   = se3_rotation_deg(make_se3(R_recovered, np.zeros(3)),
                                   make_se3(R,          np.zeros(3)))
    assert delta_deg < 1.0


class _RecordingSink:
    """VizSink double — records every on_frame call for assertions."""
    def __init__(self) -> None:
        self.frames: list[tuple[PoseResult, Keyframe | None]] = []

    def on_load(self, keyframes: list[Keyframe]) -> None:
        self.loaded = list(keyframes)

    def on_frame(self, image_rgb, geom, result, new_keyframe):
        self.frames.append((result, new_keyframe))


class _CalibratingGeometry(_FakeGeometry):
    """Drop-in `_FakeGeometry` that mimics MoGe's calibration window."""
    def __init__(self, *args, calibration_frames: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._left = calibration_frames

    @property
    def is_calibrated(self) -> bool:
        return self._left <= 0

    def __call__(self, image_rgb):
        if self._left > 0:
            self._left -= 1
        return super().__call__(image_rgb)


def test_localizer_defers_origin_until_calibrated(tmp_path: pathlib.Path):
    """While the geometry backend reports `is_calibrated=False` the localizer
    must stay in `calibrating` and never create the origin keyframe — that
    would anchor the world frame to a wrong intrinsic.

    The localizer now checks ``is_calibrated`` *before* running geometry,
    so the first N calls all return ``state="calibrating"`` (samples 1..N
    are still being collected at the start of each).  The (N+1)th call
    starts with the backend already pinned and seeds the origin.
    """
    store = KeyframeStore(tmp_path)
    geom  = _CalibratingGeometry(calibration_frames=3)
    feats = _FakeFeatures(W=geom.W, H=geom.H)
    loc   = Localizer(store=store, geometry=geom, features=feats, min_inliers=10)

    img = _white_image(geom.W, geom.H)
    for i in range(3):
        r = loc.process(img, ts_us=i)
        assert r.state == "calibrating", f"frame {i}: expected calibrating, got {r.state}"
        assert r.num_keyframes == 0
        assert r.pose is None

    # Fourth call: backend already calibrated → origin seeded, state=empty.
    r4 = loc.process(img, ts_us=100)
    assert r4.state == "empty"
    assert r4.num_keyframes == 1

    # Fifth call should localize against the now-existing origin keyframe.
    r5 = loc.process(img, ts_us=200)
    assert r5.state == "localized"


def test_localizer_emits_viz_events(tmp_path: pathlib.Path):
    """The viz sink must see the origin insertion, then a localized frame
    against that origin, and exceptions from the sink must not break
    localization."""
    store = KeyframeStore(tmp_path)
    geom  = _FakeGeometry()
    feats = _FakeFeatures(W=geom.W, H=geom.H)
    sink  = _RecordingSink()
    loc   = Localizer(store=store, geometry=geom, features=feats,
                      min_inliers=10, viz=sink)

    img = _white_image(geom.W, geom.H)
    r1 = loc.process(img, ts_us=1)
    r2 = loc.process(img, ts_us=2)

    # Both calls fired the sink; first was the origin insertion, second was
    # a pure localization (no new keyframe because pose didn't drift).
    assert len(sink.frames) == 2
    res1, kf1 = sink.frames[0]
    res2, kf2 = sink.frames[1]
    assert res1.state == "empty" and kf1 is not None and kf1.id == 0
    assert res2.state == "localized" and kf2 is None

    # A sink that raises must not interrupt the localizer (viewer is debug
    # UI; localization is the contract).
    class _Boom:
        def on_load(self, _): pass
        def on_frame(self, *args, **kwargs): raise RuntimeError("kapow")
    loc._viz = _Boom()
    r3 = loc.process(img, ts_us=3)
    assert r3.state == "localized"


def test_pose_graph_loop_closure_corrects_drift(tmp_path: pathlib.Path):
    """A four-keyframe ring (kf0..kf3) where the sequential chain disagrees
    with a kf3→kf0 loop-closure edge.  Run PGO and verify the optimizer
    redistributes the drift so kf3's optimized world pose is closer to the
    loop-closure prediction than to the chain-only prediction."""
    pg = PoseGraph(tmp_path / "edges.jsonl")

    # Truth: identity ring at 1 m spacing along +X — kf0 at origin, kf1 at
    # +1 X, kf2 at +1 X again, kf3 at +1 X.  But the chain "drifts" by
    # +0.3 m on Y between kf2 and kf3 (a mistake), so the chain-only kf3
    # pose is at (3, 0.3, 0).  The loop closure observes kf3 -> kf0 directly
    # and says they're 3 m apart on X with no Y offset.
    I = np.eye(4)
    def T(x, y, z): return make_se3(np.eye(3), [x, y, z])

    kf_poses = {
        0: T(0, 0, 0),
        1: T(1, 0, 0),
        2: T(2, 0, 0),
        3: T(3, 0.3, 0),   # drifted
    }

    # Chain edges (use what the drifted poses imply).
    pg.add(PoseEdge(0, 1, invert_se3(kf_poses[0]) @ kf_poses[1], inliers=200))
    pg.add(PoseEdge(1, 2, invert_se3(kf_poses[1]) @ kf_poses[2], inliers=200))
    pg.add(PoseEdge(2, 3, invert_se3(kf_poses[2]) @ kf_poses[3], inliers=200))

    # Loop closure: kf3 sees kf0 (or vice versa) without the Y drift.
    pg.add(PoseEdge(3, 0, invert_se3(T(3, 0, 0)) @ T(0, 0, 0),
                    inliers=300, kind="loop"))

    new_poses, info = pg.optimize(kf_poses, origin_id=0)
    assert info["final_err"] < info["initial_err"]

    # kf3's optimized Y should be much closer to 0 than to 0.3.
    drifted_y = float(kf_poses[3][1, 3])
    fixed_y   = float(new_poses[3][1, 3])
    assert abs(fixed_y) < abs(drifted_y) * 0.5, (
        f"PGO didn't redistribute drift: kf3 Y went from {drifted_y:.3f} → {fixed_y:.3f}"
    )

    # Origin must not have moved (hard prior).
    np.testing.assert_allclose(new_poses[0], kf_poses[0], atol=1e-6)


def test_inliers_persist_with_keyframe(tmp_path: pathlib.Path):
    """Each keyframe records the PnP inlier count it was anchored with —
    the viz sink filters on this to skip marginal reconstructions, so
    the value has to survive a round trip through the JSONL."""
    store = KeyframeStore(tmp_path)
    args = _fake_kf_args(1)
    store.append(**args, inliers=237)
    store.append(**_fake_kf_args(2))   # default 0

    reloaded = KeyframeStore(tmp_path)
    assert reloaded.all()[0].inliers == 237
    assert reloaded.all()[1].inliers == 0


def test_localizer_downsamples_input(tmp_path: pathlib.Path):
    """When ``image_max_edge`` is set, the localizer must shrink the
    incoming image before it ever reaches the geometry or feature
    backends — that's where the speed win comes from."""
    seen_shapes: list[tuple[int, int]] = []

    class _RecordingGeom(_FakeGeometry):
        def __call__(self, image_rgb):
            seen_shapes.append(image_rgb.shape[:2])
            return super().__call__(image_rgb)

    store = KeyframeStore(tmp_path)
    geom  = _RecordingGeom(W=320, H=240)   # backend's own resolution
    feats = _FakeFeatures(W=geom.W, H=geom.H)
    loc   = Localizer(store=store, geometry=geom, features=feats,
                      min_inliers=10, image_max_edge=128)

    big_img = np.zeros((480, 640, 3), dtype=np.uint8)
    loc.process(big_img, ts_us=1)
    # 640x480 with max_edge=128 → scale = 0.2 → 128x96.
    assert seen_shapes[0] == (96, 128)


def test_pose_graph_edges_persist_across_restart(tmp_path: pathlib.Path):
    """Edges written by one PoseGraph instance must reload on the next."""
    edges_path = tmp_path / "edges.jsonl"
    pg = PoseGraph(edges_path)
    pg.add(PoseEdge(0, 1, make_se3(np.eye(3), [1.0, 0.0, 0.0]), inliers=42))
    pg.add(PoseEdge(1, 2, make_se3(np.eye(3), [0.0, 1.0, 0.0]),
                    inliers=100, kind="loop"))
    del pg

    pg2 = PoseGraph(edges_path)
    assert len(pg2) == 2
    e0, e1 = pg2.edges()
    assert (e0.src_id, e0.dst_id, e0.kind, e0.inliers) == (0, 1, "chain", 42)
    assert (e1.src_id, e1.dst_id, e1.kind, e1.inliers) == (1, 2, "loop", 100)
    np.testing.assert_allclose(e0.rel_pose[:3, 3], [1.0, 0.0, 0.0])


def test_localizer_persists_across_restart(tmp_path: pathlib.Path):
    geom  = _FakeGeometry()
    feats = _FakeFeatures(W=geom.W, H=geom.H)
    img   = _white_image(geom.W, geom.H)

    store = KeyframeStore(tmp_path)
    loc   = Localizer(store=store, geometry=geom, features=feats, min_inliers=10)
    loc.process(img, ts_us=1)
    loc.process(img, ts_us=2)
    n_before = len(store)
    assert n_before >= 1

    # Simulate restart: drop the old store reference, rebuild from disk.
    del store, loc
    store2 = KeyframeStore(tmp_path)
    assert len(store2) == n_before
    loc2 = Localizer(store=store2, geometry=geom, features=feats, min_inliers=10)
    r = loc2.process(img, ts_us=3)
    assert r.state == "localized"
    assert r.num_keyframes == n_before
