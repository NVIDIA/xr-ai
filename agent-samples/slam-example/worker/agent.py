# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SlamAgent — space-mcp worker (topological place memory).

This branch wires space-mcp, which embeds each frame with DINOv2 and
classifies it into a topological region.  There is no metric pose to
emit and no IMU / intrinsics input — DINOv2 doesn't care.

Inputs (from the LiveKit client via xr-media-hub)
-------------------------------------------------
* FrameSignal      — per-frame metadata; pull pixels with `request_frame`.
* data topic `imu`         — ignored on this branch (space-mcp has no IMU).
* data topic `camera_meta` — logged once per participant; no intrinsics push.

Outputs
-------
* data topic `place.update` — JSON payload echoing space-mcp's
                              ``process_frame`` result.  Fields:
                              ``source``, ``state``, ``region_id``,
                              ``region_name``, ``confidence``,
                              ``num_regions``, ``transitioned_from``,
                              ``ts_us``.

Frames are resized to a fixed longest-edge tracking size before being
pushed to space-mcp.  DINOv2 internally rescales to 224 px anyway, so
bounding the input image here just keeps the on-wire PNG small.
"""
from __future__ import annotations

import asyncio
import json
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

        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown = False

    # ── data-channel ingest ─────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic == "camera_meta":
            self._log_camera_meta(msg.participant_id, msg.data)
            return
        # ``imu`` and any other topics are no-ops on this branch.
        # space-mcp does pure visual place recognition with DINOv2 —
        # no IMU fusion, no intrinsics needed.

    def _log_camera_meta(self, pid: str, payload: bytes) -> None:
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

    def _track_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        """Longest-edge=``slam_track_max_edge`` while preserving source
        aspect ratio.  Rounded to even pixels."""
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
        """Wake on each new frame, throttle to ``slam_hz``, call
        process_frame, echo the region info back."""
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

        t0 = time.monotonic()
        try:
            result = await self._slam.process_frame(
                str(out_path), timestamp_us=frame.pts_us,
            )
        except Exception as exc:
            logger.warning("slam process_frame failed pid={!r}: {}", pid, exc)
            return
        self._last_pts_per_pid[pid] = sig.pts_us
        dt_ms = (time.monotonic() - t0) * 1000.0

        if result.get("error"):
            logger.warning("slam-mcp error pid={!r}: {}", pid, result["error"])
            return

        state        = result.get("state")
        region_id    = result.get("region_id")
        region_name  = result.get("region_name")
        confidence   = result.get("confidence")
        num_regions  = result.get("num_regions")
        trans_from   = result.get("transitioned_from")
        ts_us        = result.get("ts_us")

        logger.info(
            "slam  pid={!r}  state={}  region={} ({!r})  conf={}  regions={}  ({:.0f} ms)",
            pid, state, region_id, region_name,
            f"{confidence:.3f}" if isinstance(confidence, (int, float)) else confidence,
            num_regions, dt_ms,
        )
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="place.update",
            pts_us=frame.pts_us,
            data=json.dumps({
                "source":            "space-mcp",
                "state":             state,
                "region_id":         region_id,
                "region_name":       region_name,
                "confidence":        confidence,
                "num_regions":       num_regions,
                "transitioned_from": trans_from,
                "ts_us":             ts_us,
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
