# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SlamAgent — SLAM-only worker.

Inputs (from the LiveKit client via xr-media-hub)
-------------------------------------------------
* FrameSignal      — per-frame metadata; pull pixels with `request_frame`.
* data topic `imu`         — batched DeviceMotionEvent rows, packed as JSON.
* data topic `camera_meta` — once per startCamera, payload describes the
                              camera that's now streaming.

Outputs
-------
* data topic `pose.update` — JSON payload with `source`, `state`,
                              `translation_m`, `quaternion`, `frames_sent`,
                              `ts_ns`.

The worker resizes every frame to a fixed longest-edge tracking size
(default 320 px) before pushing it to the SLAM MCP server.  LiveKit
can change a track's resolution mid-session (simulcast / adaptive
bitrate) and several SLAM backends crash if the input resolution
changes after init — fixing the tracking size at the worker layer
sidesteps that entirely.  Intrinsics pushed via ``set_camera_intrinsics``
are for the tracking resolution, not the source camera's native size.
"""
from __future__ import annotations

import asyncio
import json
import math
import pathlib
import re
import time

from loguru import logger
from PIL import Image
from xr_ai_agent import DataMessage, FrameSignal, ProcessorEndpoint

from pixels      import frame_to_pil
from slam_client import SlamClient


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _safe_pid(pid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", pid) or "anon"


def _fmt3(v) -> str:
    if v is None:
        return "—"
    try:
        return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"
    except Exception:
        return str(v)


def _guess_fov_from_meta(meta: dict) -> float | None:
    """Phone cameras typically expose horizontal FOV in EXIF and the
    WebRTC ``facingMode`` flag.  We use a small lookup with sensible
    defaults — better than guessing a wildly wrong number on every
    first frame.  Override via ``set_camera_intrinsics`` if you know
    the real FOV.
    """
    fov = meta.get("fov_x")
    if fov:
        return float(fov)
    label   = (meta.get("label") or "").lower()
    facing  = (meta.get("facing") or "").lower()
    # Most front cameras on phones are wider than the rear ones.
    if "front" in label or facing == "user":
        return 70.0
    if "back" in label or "rear" in label or facing == "environment":
        return 65.0
    return 60.0


class SlamAgent:

    def __init__(
        self,
        ep:    ProcessorEndpoint,
        slam:  SlamClient,
        *,
        slam_hz:           float = 2.0,
        slam_max_age_s:    float = 1.0,
        slam_scratch_dir:  pathlib.Path = pathlib.Path("/dev/shm/xr-ai/slam-in"),
        slam_track_max_edge: int = 320,
    ) -> None:
        self._ep    = ep
        self._slam  = slam

        self._ep.on_data(self._on_data)
        self._ep.on_frame(self._on_frame)

        self._min_period_s   = 1.0 / max(slam_hz, 0.1)
        self._max_age_s      = float(slam_max_age_s)
        self._scratch_dir    = slam_scratch_dir
        self._track_max_edge = max(64, int(slam_track_max_edge))

        self._latest:           dict[tuple[str, str], FrameSignal] = {}
        self._last_pts_per_pid: dict[str, int]                     = {}
        self._event             = asyncio.Event()
        self._imu_pending:      list[list[float]]                  = []
        self._intrinsics_set:   set[str]                           = set()

        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown = False

    # ── data-channel ingest ─────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic == "imu":
            self._ingest_imu(msg.data)
            return
        if msg.topic == "camera_meta":
            self._ingest_camera_meta(msg.participant_id, msg.data)
            return
        # Any other text payload is ignored — this sample doesn't have
        # a query path (no VLM / STT / TTS).  Use simple-vlm-example
        # if you want voice / text queries.

    def _ingest_imu(self, payload: bytes) -> None:
        """Decode the web client's batched DeviceMotionEvent payload and
        stash the raw rows for the next ``push_imu`` call.

        Payload shape::

            {"t": <ms>, "dt": <ms>, "a": [[ax,ay,az], ...],
                                    "alin": [[ax,ay,az], ...],
                                    "g":    [[gx,gy,gz], ...]}
        """
        try:
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            return
        gs   = data.get("g")    or []
        accs = data.get("alin") or data.get("a") or []
        if not gs or not accs:
            return
        t_ms  = float(data.get("t",  0))
        dt_ms = float(data.get("dt", 0))
        for i, (g, a) in enumerate(zip(gs, accs)):
            try:
                self._imu_pending.append([
                    t_ms + i * dt_ms,
                    float(g[0]), float(g[1]), float(g[2]),
                    float(a[0]), float(a[1]), float(a[2]),
                ])
            except (TypeError, ValueError, IndexError):
                continue

    def _ingest_camera_meta(self, pid: str, payload: bytes) -> None:
        try:
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            return
        logger.info(
            "camera meta  pid={!r}  {}x{}  fps={}  facing={}  label={!r}",
            pid, data.get("width"), data.get("height"),
            data.get("frame_rate"), data.get("facing"),
            (data.get("label") or "")[:50],
        )
        if pid not in self._intrinsics_set:
            asyncio.create_task(self._push_intrinsics(pid, data))

    async def _push_intrinsics(self, pid: str, meta: dict) -> None:
        """Derive pinhole K for the *tracking* resolution and push it.
        Since every frame gets resized to ``slam_track_max_edge`` before
        being sent, intrinsics are computed for that size — the source
        resolution from camera_meta is informational only."""
        src_w = int(meta.get("width")  or 0)
        src_h = int(meta.get("height") or 0)
        if src_w <= 0 or src_h <= 0:
            return
        fov_x = _guess_fov_from_meta(meta) or 60.0
        track_w, track_h = self._track_size(src_w, src_h)
        fx = 0.5 * track_w / math.tan(0.5 * math.radians(fov_x))
        try:
            r = await self._slam.set_camera_intrinsics(
                width=track_w, height=track_h, fx=fx, fy=fx,
                cx=track_w / 2.0, cy=track_h / 2.0,
            )
            logger.info(
                "slam intrinsics set  source {}x{} → tracked {}x{}  "
                "fx={:.1f} ({:.0f}° FOV)  resp={}",
                src_w, src_h, track_w, track_h, fx, fov_x, r,
            )
            self._intrinsics_set.add(pid)
        except Exception as exc:
            logger.warning("slam set_camera_intrinsics failed: {}", exc)

    def _track_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        """Longest-edge=``slam_track_max_edge`` while preserving source
        aspect ratio.  Rounded to even pixels so downstream PIL /
        encoder paths stay happy."""
        m = self._track_max_edge
        if src_w >= src_h:
            w = m
            h = max(2, int(round(src_h * (m / src_w))))
        else:
            h = m
            w = max(2, int(round(src_w * (m / src_h))))
        return (w & ~1, h & ~1)

    # ── frame-signal ingest ─────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._latest[(sig.participant_id, sig.track_id)] = sig
        self._event.set()

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        best: FrameSignal | None = None
        for (p, _), s in self._latest.items():
            if p != pid:
                continue
            if best is None or s.pts_us > best.pts_us:
                best = s
        return best

    # ── slam loop ───────────────────────────────────────────────────────

    async def _slam_loop(self) -> None:
        """Wake on each new frame, throttle to ``slam_hz``, flush
        buffered IMU, call estimate_pose, echo the pose back."""
        logger.info(
            "slam loop running  min_period={:.2f}s  max_age={:.2f}s  scratch={}  track_max={}",
            self._min_period_s, self._max_age_s,
            self._scratch_dir, self._track_max_edge,
        )
        idle_logged = False
        try:
            while not self._shutdown:
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                self._event.clear()
                pids = {pid for pid, _ in self._latest}
                if not pids:
                    if not idle_logged:
                        logger.info("slam loop idle — no participants with frames yet")
                        idle_logged = True
                    continue
                idle_logged = False
                for pid in pids:
                    try:
                        await self._slam_one(pid)
                    except Exception as exc:
                        logger.opt(exception=True).warning(
                            "slam iteration failed pid={!r}: {}", pid, exc,
                        )
                await asyncio.sleep(self._min_period_s)
        except asyncio.CancelledError:
            raise

    async def _slam_one(self, pid: str) -> None:
        sig = self._latest_signal(pid)
        if sig is None:
            return
        age_s = (_now_us() - sig.pts_us) / 1_000_000.0
        if age_s > self._max_age_s:
            return
        if self._last_pts_per_pid.get(pid) == sig.pts_us:
            return
        frame = await self._ep.request_frame(sig)
        if frame is None:
            return

        img = frame_to_pil(frame)
        track_w, track_h = self._track_size(img.width, img.height)
        if (img.width, img.height) != (track_w, track_h):
            img = img.resize((track_w, track_h), Image.Resampling.BILINEAR)
        out_path = self._scratch_dir / f"{_safe_pid(pid)}.png"
        tmp_path = out_path.with_suffix(".png.tmp")
        img.save(tmp_path, format="PNG")
        tmp_path.replace(out_path)

        # Flush IMU buffer.
        if self._imu_pending:
            batch = self._imu_pending
            self._imu_pending = []
            try:
                await self._slam.push_imu(batch)
            except Exception as exc:
                logger.warning("slam push_imu failed: {}", exc)

        t0 = time.monotonic()
        try:
            result = await self._slam.estimate_pose(
                str(out_path), timestamp_us=frame.pts_us,
            )
        except Exception as exc:
            logger.warning("slam estimate_pose failed pid={!r}: {}", pid, exc)
            return
        self._last_pts_per_pid[pid] = sig.pts_us
        dt_ms = (time.monotonic() - t0) * 1000.0

        if result.get("error"):
            logger.warning("slam-mcp error pid={!r}: {}", pid, result["error"])
            return

        state = result.get("state")
        t     = result.get("translation_m")
        logger.info(
            "slam  pid={!r}  state={}  t={}  frames={}  ({:.0f} ms)",
            pid, state, _fmt3(t),
            result.get("frames_sent"), dt_ms,
        )
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="pose.update",
            pts_us=frame.pts_us,
            data=json.dumps({
                "source":        "slam",
                "state":         state,
                "translation_m": t,
                "quaternion":    result.get("quaternion"),
                "frames_sent":   result.get("frames_sent"),
                "ts_ns":         result.get("ts_ns"),
            }).encode(),
        ))

    # ── lifecycle ───────────────────────────────────────────────────────

    async def run(self) -> None:
        loop = asyncio.create_task(self._slam_loop(), name="slam-loop")
        try:
            await loop
        except asyncio.CancelledError:
            pass

    def shutdown(self) -> None:
        self._shutdown = True
        self._event.set()
