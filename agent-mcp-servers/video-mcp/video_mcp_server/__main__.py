# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Video MCP server.

Pure FastMCP — every operation is an MCP tool at /mcp. There are no REST
endpoints. Use ``fastmcp.Client`` (or any MCP client) to query.

Two data paths:

* **Historical chunks** — reads the H.264 Annex B chunks the hub recorder
  writes to (tmpfs by default). Used by ``query_video``,
  ``get_video_stats``, ``list_recorded_participants``,
  ``get_frame_at_time``.

* **Live frames** — connects to the hub as a ``ProcessorEndpoint``,
  tracks the most recent ``FrameSignal`` per participant, and pulls
  pixels on demand via ``request_frame``. Used by
  ``list_live_participants`` and ``get_latest_frame``.

All tools accept and return raw LiveKit identities; sanitization happens
internally for filesystem paths and is recovered via ``.identity``
sidecars written by the recorder.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  list_live_participants() → list[str]
      Identities currently connected to the hub (live IPC roster).

  list_recorded_participants() → list[str]
      Identities that have at least one chunk on disk.

  get_video_stats(participant_id) → dict
      num_chunks, total_bytes, avg_chunk_bytes, earliest_us, latest_us.

  query_video(participant_id, start_us, end_us) → dict
      Concatenate H.264 chunks overlapping the window, write to a file,
      return the path. Result is raw H.264 starting with an IDR.

  get_latest_frame(participant_id) → dict
      Fetch the most recent frame the hub has for *participant_id*,
      encode to PNG, return the file path. Uses the live IPC path —
      independent of disk chunk timing.

  get_frame_at_time(participant_id, timestamp_us) → dict
      Decode the chunk covering *timestamp_us*, pick the frame closest
      to that timestamp, encode to PNG, return the file path.

Config (video_mcp_server.yaml)
───────────────────────────────
    recordings_dir: /dev/shm/xr-ai/recordings   # must match hub video_recording.out_dir
    out_dir:        /tmp/xr_video_queries
    hub_pub:        ipc:///tmp/xr_hub_pub        # hub PUB socket (live frames)
    hub_push:       ipc:///tmp/xr_hub_in         # hub PUSH socket (frame requests)
    host:           0.0.0.0
    port:           8210
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib

import numpy as np
import uvicorn
import yaml
from fastmcp import FastMCP
from PIL import Image

from xr_ai_agent import (FrameData, FrameSignal, PixelFormat,
                         ProcessorEndpoint, Subscribe)

log = logging.getLogger("video_mcp_server")

_DEFAULT_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_DEFAULT_HUB_PUSH = "ipc:///tmp/xr_hub_in"


def _safe_name(s: str) -> str:
    """Filesystem-safe version of *s*. Mirrors the recorder's helper."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# ── chunk store (reads the recording directory) ───────────────────────────────

class ChunkStore:
    def __init__(self, recordings_dir: pathlib.Path) -> None:
        self._root = recordings_dir

    def _pid_dir(self, pid: str) -> pathlib.Path | None:
        """Return the existing dir whose ``.identity`` matches *pid*, or
        ``None`` if no recordings exist for that participant."""
        if not self._root.exists():
            return None
        safe = _safe_name(pid)
        # Fast path: canonical name and its .identity matches (or, for
        # legacy pre-sidecar dirs, the dir name == raw == safe).
        canonical = self._root / safe
        if canonical.is_dir():
            sidecar = canonical / ".identity"
            if sidecar.exists():
                if sidecar.read_text(encoding="utf-8") == pid:
                    return canonical
            elif pid == safe:
                return canonical
        # Slow path: scan all dirs (covers collision-bumped suffixes).
        for d in sorted(self._root.iterdir()):
            if not d.is_dir():
                continue
            sidecar = d / ".identity"
            if sidecar.exists() and sidecar.read_text(encoding="utf-8") == pid:
                return d
        return None

    def list_participants(self) -> list[str]:
        """Return raw participant identities for every recorded
        participant (read from ``.identity`` sidecars; falls back to the
        directory name for legacy dirs without a sidecar)."""
        if not self._root.exists():
            return []
        out: list[str] = []
        for d in sorted(self._root.iterdir()):
            if not d.is_dir():
                continue
            sidecar = d / ".identity"
            if sidecar.exists():
                out.append(sidecar.read_text(encoding="utf-8"))
            else:
                out.append(d.name)
        return out

    def _sorted_chunks(self, pid: str) -> list[pathlib.Path]:
        pid_dir = self._pid_dir(pid)
        if pid_dir is None:
            return []
        return sorted(pid_dir.glob("*.264"), key=lambda p: int(p.stem))

    def _load_meta(self, h264: pathlib.Path) -> dict:
        meta_path = h264.with_suffix(".json")
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        # Fall back to filename stem if sidecar is missing.
        return {"start_us": int(h264.stem), "end_us": int(h264.stem), "size_bytes": h264.stat().st_size}

    def stats(self, pid: str) -> dict | None:
        chunks = self._sorted_chunks(pid)
        if not chunks:
            return None
        metas      = [self._load_meta(c) for c in chunks]
        total      = sum(m.get("size_bytes", c.stat().st_size) for m, c in zip(metas, chunks))
        return {
            "participant_id":  pid,
            "num_chunks":      len(chunks),
            "total_bytes":     total,
            "avg_chunk_bytes": total // len(chunks),
            "earliest_us":     metas[0].get("start_us",  int(chunks[0].stem)),
            "latest_us":       metas[-1].get("end_us",   int(chunks[-1].stem)),
        }

    def query(self, pid: str, start_us: int, end_us: int) -> bytes | None:
        chunks = self._sorted_chunks(pid)
        if not chunks:
            return None

        metas = [(c, self._load_meta(c)) for c in chunks]

        # Include the last chunk that started before the window (gives us an IDR
        # at the start of the result) plus all chunks that overlap the window.
        anchor  = None
        overlap = []
        for h264, meta in metas:
            cs = meta.get("start_us", int(h264.stem))
            ce = meta.get("end_us",   cs)
            if cs <= end_us and ce >= start_us:
                overlap.append(h264)
            elif cs < start_us:
                anchor = h264   # latest chunk entirely before the window

        selected = ([anchor] if anchor and not overlap else []) + overlap
        if not selected:
            return None

        return b"".join(p.read_bytes() for p in selected)

    def find_chunk_at(self, pid: str, ts_us: int) -> tuple[pathlib.Path, dict] | None:
        """Return the chunk whose [start_us, end_us] window contains
        *ts_us*, or the chunk whose start is closest to *ts_us* if none
        contains it. None if no chunks exist for *pid*."""
        chunks = self._sorted_chunks(pid)
        if not chunks:
            return None
        metas = [(c, self._load_meta(c)) for c in chunks]
        for c, m in metas:
            if m.get("start_us", int(c.stem)) <= ts_us <= m.get("end_us", int(c.stem)):
                return c, m
        # Fall through: pick the chunk whose start is closest.
        best = min(metas, key=lambda cm: abs(cm[1].get("start_us", int(cm[0].stem)) - ts_us))
        return best


# ── live frame provider (ProcessorEndpoint) ───────────────────────────────────

class FrameProvider:
    """Tracks the most recent ``FrameSignal`` per participant via IPC.

    The hub publishes frame metadata on every frame; we keep the latest
    signal for each pid. ``fetch_latest`` issues a ``FRAME_REQUEST`` to
    pull the actual pixel bytes on demand — the hub copies from the SHM
    slot only when asked.

    Subscribed with ``filter=Subscribe.VIDEO`` so we don't pay the SUB-
    side decode cost for audio / data we don't care about.
    """

    def __init__(self, ep: ProcessorEndpoint) -> None:
        self._ep = ep
        self._latest: dict[str, FrameSignal] = {}
        ep.on_frame(self._on_frame)

    async def _on_frame(self, sig: FrameSignal) -> None:
        # Take the most recent signal across all of the pid's tracks.
        prev = self._latest.get(sig.participant_id)
        if prev is None or sig.pts_us >= prev.pts_us:
            self._latest[sig.participant_id] = sig

    def latest_signal(self, pid: str) -> FrameSignal | None:
        return self._latest.get(pid)

    def connected_participants(self) -> frozenset[str]:
        """Raw identities currently connected to the hub (live IPC roster)."""
        return self._ep.connected_participants

    async def fetch_latest(self, pid: str) -> FrameData | None:
        sig = self._latest.get(pid)
        if sig is None:
            return None
        return await self._ep.request_frame(sig)


# ── pixel format conversion ───────────────────────────────────────────────────

def _frame_to_rgb(data: bytes, width: int, height: int, fmt: PixelFormat) -> np.ndarray:
    """Convert a hub ``FrameData`` payload into an HxWx3 uint8 RGB array."""
    arr = np.frombuffer(data, dtype=np.uint8)

    if fmt == PixelFormat.RGB24:
        return arr.reshape(height, width, 3).copy()
    if fmt == PixelFormat.RGBA:
        return arr.reshape(height, width, 4)[:, :, :3].copy()
    if fmt == PixelFormat.BGRA:
        bgra = arr.reshape(height, width, 4)
        return bgra[:, :, [2, 1, 0]].copy()
    if fmt == PixelFormat.NV12:
        return _nv12_to_rgb(arr.reshape(height * 3 // 2, width), width, height)
    if fmt == PixelFormat.I420:
        y_size  = width * height
        uv_size = (width // 2) * (height // 2)
        Y = arr[:y_size].reshape(height, width)
        U = arr[y_size : y_size + uv_size].reshape(height // 2, width // 2)
        V = arr[y_size + uv_size :].reshape(height // 2, width // 2)
        return _yuv_to_rgb(Y, U, V, width, height)

    raise ValueError(f"Unsupported PixelFormat for PNG export: {fmt!r}")


def _nv12_to_rgb(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    """NV12 (Y plane + interleaved Cb/Cr at half resolution) → RGB."""
    Y  = nv12[:height, :].astype(np.float32)
    UV = nv12[height:, :].reshape(height // 2, width // 2, 2)
    Cb = np.repeat(np.repeat(UV[:, :, 0], 2, axis=0), 2, axis=1).astype(np.float32)
    Cr = np.repeat(np.repeat(UV[:, :, 1], 2, axis=0), 2, axis=1).astype(np.float32)
    return _yuv_arr_to_rgb(Y, Cb, Cr)


def _yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                width: int, height: int) -> np.ndarray:
    Y_f  = Y.astype(np.float32)
    Cb_f = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1).astype(np.float32)
    Cr_f = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1).astype(np.float32)
    return _yuv_arr_to_rgb(Y_f, Cb_f, Cr_f)


def _yuv_arr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """BT.601 limited-range YUV → RGB. Inverse of the recorder's RGB→YCbCr."""
    Y  = Y  - 16
    Cb = Cb - 128
    Cr = Cr - 128
    R = 1.164 * Y + 1.596 * Cr
    G = 1.164 * Y - 0.392 * Cb - 0.813 * Cr
    B = 1.164 * Y + 2.017 * Cb
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _save_png(rgb: np.ndarray, out_path: pathlib.Path) -> None:
    Image.fromarray(rgb, "RGB").save(out_path, "PNG")


# ── H.264 decode (PyNvVideoCodec) ────────────────────────────────────────────

def _decode_chunk_to_nv12_frames(annex_b: bytes, gpu_id: int = 0) -> list[np.ndarray]:
    """Decode an H.264 Annex B chunk into a list of NV12 numpy arrays.

    Each returned array has shape ``(H*3//2, W)`` (the same layout the
    encoder consumes). Uses NVDEC via PyNvVideoCodec — no software
    fallback.
    """
    import PyNvVideoCodec as nvc
    decoder = nvc.CreateDecoder(
        gpuid=gpu_id, codec="h264", cudacontext=0, cudastream=0,
        usedevicememory=False,
    )
    frames: list[np.ndarray] = []
    for frame in decoder.Decode(annex_b):
        # `frame` exposes the NumPy buffer protocol; copy out so it
        # outlives the decoder's internal slot.
        frames.append(np.array(frame, copy=True))
    return frames


# ── server ────────────────────────────────────────────────────────────────────

def build_mcp(
    store:    ChunkStore,
    out_dir:  pathlib.Path,
    provider: FrameProvider,
    gpu_id:   int = 0,
) -> "FastMCP":
    """Return a composed FastMCP server with all video tools bound."""
    mcp = FastMCP("video-mcp")

    @mcp.tool()
    def list_live_participants() -> list[str]:
        """Return raw participant identities currently connected to the
        hub. Drawn from the live IPC roster — these are the only pids
        for which ``get_latest_frame`` will return a frame."""
        return sorted(provider.connected_participants())

    @mcp.tool()
    def list_recorded_participants() -> list[str]:
        """Return raw participant identities that have at least one
        recorded chunk on disk. Read from ``.identity`` sidecars; covers
        both currently-connected and previously-connected participants
        whose chunks are still within the recorder's eviction window."""
        return store.list_participants()

    @mcp.tool()
    def get_video_stats(participant_id: str) -> dict:
        """
        Summary statistics for all recorded chunks of *participant_id*.

        Keys: participant_id, num_chunks, total_bytes, avg_chunk_bytes,
              earliest_us (Unix µs), latest_us (Unix µs).
        Returns an error dict if no chunks exist.
        """
        result = store.stats(participant_id)
        if result is None:
            return {"error": f"No video chunks for {participant_id!r}"}
        return result

    @mcp.tool()
    def query_video(participant_id: str, start_us: int, end_us: int) -> dict:
        """
        Concatenate H.264 chunks for *participant_id* covering [start_us, end_us]
        (Unix microseconds), write to a file, and return the path.

        The result is a raw H.264 Annex B stream starting with an IDR frame.
        Keys: path (str), size (int), start_us (int), end_us (int).
        """
        data = store.query(participant_id, start_us, end_us)
        if data is None:
            return {"error": f"No video chunks for {participant_id!r} in requested window"}
        safe = _safe_name(participant_id)
        out_path = out_dir / f"{safe}_{start_us}_{end_us}.264"
        out_path.write_bytes(data)
        log.info("query_video  pid=%r  %d–%d  %d bytes → %s",
                 participant_id, start_us, end_us, len(data), out_path)
        return {"path": str(out_path), "size": len(data),
                "start_us": start_us, "end_us": end_us}

    @mcp.tool()
    async def get_latest_frame(participant_id: str) -> dict:
        """
        Fetch the most recent frame the hub has for *participant_id*,
        encode to PNG, return the file path.

        Keys: path, width, height, timestamp_us, track_id.
        Returns an error dict if no live frame is available yet.
        """
        frame = await provider.fetch_latest(participant_id)
        if frame is None:
            return {"error": f"No live frame available for {participant_id!r}"}
        try:
            rgb = _frame_to_rgb(frame.data, frame.width, frame.height, frame.fmt)
        except ValueError as exc:
            return {"error": str(exc)}
        safe     = _safe_name(participant_id)
        out_path = out_dir / f"{safe}_latest_{frame.pts_us}.png"
        _save_png(rgb, out_path)
        log.info("get_latest_frame  pid=%r  %dx%d  ts=%d → %s",
                 participant_id, frame.width, frame.height, frame.pts_us, out_path)
        return {
            "path":         str(out_path),
            "width":        frame.width,
            "height":       frame.height,
            "timestamp_us": frame.pts_us,
            "track_id":     frame.track_id,
        }

    @mcp.tool()
    def get_frame_at_time(participant_id: str, timestamp_us: int) -> dict:
        """
        Decode the H.264 chunk covering *timestamp_us* (Unix µs) for
        *participant_id*, pick the frame closest to that timestamp,
        encode to PNG, return the file path.

        Keys: path, width, height, timestamp_us (the actual frame ts,
        approximated by linear interpolation within the chunk), chunk_path.
        Returns an error dict if no chunks cover the request.
        """
        found = store.find_chunk_at(participant_id, timestamp_us)
        if found is None:
            return {"error": f"No video chunks recorded for {participant_id!r}"}
        chunk_path, meta = found
        try:
            frames = _decode_chunk_to_nv12_frames(chunk_path.read_bytes(), gpu_id=gpu_id)
        except Exception as exc:
            log.exception("decode failed  chunk=%s", chunk_path)
            return {"error": f"Decode failed: {exc}"}
        if not frames:
            return {"error": f"Chunk {chunk_path.name} decoded zero frames"}

        start_us   = int(meta.get("start_us", int(chunk_path.stem)))
        end_us     = int(meta.get("end_us",   start_us))
        num_frames = int(meta.get("num_frames", len(frames)))
        width      = int(meta.get("width",  frames[0].shape[1]))
        height_nv  = frames[0].shape[0]
        height     = int(meta.get("height", height_nv * 2 // 3))

        # Linear interpolation: which frame index is closest to ts_us?
        if num_frames <= 1 or end_us <= start_us:
            idx = 0
        else:
            ratio = (timestamp_us - start_us) / (end_us - start_us)
            idx   = max(0, min(num_frames - 1, round(ratio * (num_frames - 1))))
        idx = min(idx, len(frames) - 1)

        nv12     = frames[idx]
        rgb      = _nv12_to_rgb(nv12, width, height)
        safe     = _safe_name(participant_id)
        out_path = out_dir / f"{safe}_at_{timestamp_us}.png"
        _save_png(rgb, out_path)

        # Approximate frame timestamp (uniform spacing inside the chunk).
        frame_ts = (
            start_us + idx * (end_us - start_us) // max(num_frames - 1, 1)
            if num_frames > 1 else start_us
        )
        log.info("get_frame_at_time  pid=%r  ts=%d  frame=%d/%d → %s",
                 participant_id, timestamp_us, idx, num_frames, out_path)
        return {
            "path":         str(out_path),
            "width":        width,
            "height":       height,
            "timestamp_us": frame_ts,
            "chunk_path":   str(chunk_path),
        }

    return mcp


def build_app(
    store:    ChunkStore,
    out_dir:  pathlib.Path,
    provider: FrameProvider,
    gpu_id:   int = 0,
):
    """Return the ASGI app serving the FastMCP HTTP transport at /mcp."""
    return build_mcp(store, out_dir, provider, gpu_id=gpu_id).http_app(path="/mcp")


# ── entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: dict) -> None:
    recordings_dir = pathlib.Path(cfg.get("recordings_dir", "/dev/shm/xr-ai/recordings"))
    out_dir        = pathlib.Path(cfg.get("out_dir",        "/tmp/xr_video_queries"))
    hub_pub        = cfg.get("hub_pub",  _DEFAULT_HUB_PUB)
    hub_push       = cfg.get("hub_push", _DEFAULT_HUB_PUSH)
    host           = cfg.get("host", "0.0.0.0")
    port           = int(cfg.get("port", 8210))
    gpu_id         = int(cfg.get("gpu_id", 0))

    out_dir.mkdir(parents=True, exist_ok=True)

    store    = ChunkStore(recordings_dir)
    ep       = ProcessorEndpoint(
        sub_addr=hub_pub, push_addr=hub_push,
        filter=Subscribe.VIDEO,
    )
    provider = FrameProvider(ep)
    app      = build_app(store, out_dir, provider, gpu_id=gpu_id)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    ep_task = asyncio.create_task(ep.run(), name="video_mcp_processor")
    log.info("video-mcp-server  recordings_dir=%s  port=%d  hub_pub=%s",
             recordings_dir, port, hub_pub)
    try:
        await server.serve()
    finally:
        ep.stop()
        ep_task.cancel()
        try:
            await ep_task
        except (asyncio.CancelledError, Exception):
            pass
        ep.close()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(_serve(cfg))


if __name__ == "__main__":
    run()
