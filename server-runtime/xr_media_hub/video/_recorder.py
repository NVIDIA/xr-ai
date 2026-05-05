# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVENC video recorder for XR-Media-Hub.

Records incoming video frames per participant to H.264 Annex B chunk files.
Each chunk starts with SPS+PPS+IDR and is independently decodable.

Storage backend
---------------
Defaults to a tmpfs path (``/dev/shm/xr-ai/recordings``) so chunk writes
go to RAM, not disk. The hub holds at most ``max_total_bytes`` (default
500 MB) of chunks across all participants combined; the oldest chunks
are evicted FIFO when the budget is exceeded.

On-disk layout
--------------
    <out_dir>/<dir_name>/
        .identity         — raw participant identity (utf-8, one line)
        <start_us>.264    — raw H.264 Annex B, starts with IDR
        <start_us>.json   — chunk metadata sidecar

``<dir_name>`` is ``_safe_name(participant_id)``; if two distinct raw
identities collide on the same safe name, the second one gets a counter
suffix (``alice_home`` then ``alice_home_2``…). The ``.identity`` file
is the source of truth for the raw name; downstream consumers should
prefer it over the directory name.

Per-chunk sidecar JSON keys:
    start_us    int   chunk start time (Unix µs, same as filename stem)
    end_us      int   chunk end time (Unix µs, written when chunk is closed)
    num_frames  int   encoded frames in this chunk
    width       int   frame width in pixels
    height      int   frame height in pixels
    size_bytes  int   .264 file size

The video MCP server reads this directory directly for historical
queries. Latest-frame queries use a separate path: video-mcp connects
to the hub as a ``ProcessorEndpoint`` and pulls live frames over IPC.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xr_media_hub.ipc import SlotView

log = logging.getLogger(__name__)


# Default chunk root. /dev/shm is tmpfs on Linux (auto-mounted at boot),
# so writes here cost RAM bandwidth, not disk IO.
_DEFAULT_OUT_DIR     = "/dev/shm/xr-ai/recordings"
_DEFAULT_MAX_BYTES   = 500 * 1024 * 1024   # 500 MB across all participants


@dataclass
class VideoRecorderConfig:
    out_dir:         str   = _DEFAULT_OUT_DIR
    chunk_frames:    int   = 30                  # frames per chunk (1 s at 30 fps)
    max_total_bytes: int   = _DEFAULT_MAX_BYTES  # global cap; FIFO eviction (0 = unlimited)
    sample_fps:      float = 30.0
    bitrate:         int   = 4_000_000
    gpu_id:          int   = 0


@dataclass
class _TrackEncoder:
    lock:           threading.Lock = field(default_factory=threading.Lock)
    encoder:        object         = None
    chunk_buf:      bytearray      = field(default_factory=bytearray)
    chunk_start_us: int            = 0
    chunk_frames:   int            = 0
    width:          int            = 0
    height:         int            = 0
    out_dir:        Path           = field(default_factory=Path)
    last_ts:        float          = 0.0


class VideoRecorder:
    def __init__(self, cfg: VideoRecorderConfig) -> None:
        try:
            import PyNvVideoCodec as nvc
            self._nvc = nvc
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "PyNvVideoCodec is required for video recording. "
                "Install it with: uv sync  (in server-runtime/)"
            ) from None

        self._cfg          = cfg
        self._out_dir      = Path(cfg.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._encoders:    dict[tuple[str, str], _TrackEncoder] = {}
        self._lock         = threading.Lock()
        self._min_interval = 1.0 / max(cfg.sample_fps, 1.0)
        # Lock around global prune so two chunk-flushes can't double-evict.
        self._prune_lock   = threading.Lock()

    # ── hub callback ──────────────────────────────────────────────────────────

    async def on_frame(self, view: "SlotView") -> None:
        import asyncio
        sig  = view.signal
        data = bytes(view.data)
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._encode_frame,
                sig.participant_id, sig.track_id,
                data, sig.width, sig.height, sig.fmt,
            )
        except Exception as exc:
            log.warning("recorder  encode error pid=%r: %s", sig.participant_id, exc)

    # ── encoding (runs in thread pool) ───────────────────────────────────────

    def _encode_frame(
        self,
        pid: str, tid: str,
        data: bytes, width: int, height: int, fmt: object,
    ) -> None:
        now = time.monotonic()

        with self._lock:
            key = (pid, tid)
            if key not in self._encoders:
                self._encoders[key] = self._make_track(pid, tid, width, height)
            enc = self._encoders[key]

        with enc.lock:
            if now - enc.last_ts < self._min_interval:
                return
            enc.last_ts = now

            if width != enc.width or height != enc.height:
                log.info("recorder  resolution change %dx%d → %dx%d  pid=%r",
                         enc.width, enc.height, width, height, pid)
                # Drop the old encoder reference first so CPython's refcount
                # hits zero and the destructor (NvEncDestroyEncoder) fires
                # before CreateEncoder opens a new session.
                old_encoder = enc.encoder
                enc.encoder = None
                try:
                    flushed = old_encoder.EndEncode()
                    if flushed:
                        enc.chunk_buf.extend(flushed)
                except Exception as e:
                    log.warning("recorder  EndEncode error on resolution change pid=%r: %s", pid, e)
                del old_encoder
                self._flush_chunk(enc)
                try:
                    enc.encoder = self._create_encoder(width, height)
                except Exception as e:
                    log.warning("recorder  CreateEncoder failed pid=%r %dx%d: %s — "
                                "recording disabled for this track", pid, width, height, e)
                    # Update dimensions so we don't retry on every frame.
                    enc.width  = width
                    enc.height = height
                    return
                enc.chunk_start_us = time.time_ns() // 1_000
                enc.chunk_frames   = 0
                enc.width          = width
                enc.height         = height

            nv12 = _to_nv12(data, width, height, fmt)
            if nv12 is None:
                return

            if enc.chunk_frames >= self._cfg.chunk_frames:
                self._rotate_chunk(enc, pid)

            encoded = enc.encoder.Encode(nv12)
            if encoded:
                enc.chunk_buf.extend(encoded)

            enc.chunk_frames += 1

    def _rotate_chunk(self, enc: _TrackEncoder, pid: str) -> None:
        try:
            flushed = enc.encoder.EndEncode()
            if flushed:
                enc.chunk_buf.extend(flushed)
        except Exception as e:
            log.warning("recorder  EndEncode error pid=%r: %s", pid, e)
        self._flush_chunk(enc)
        enc.encoder        = self._create_encoder(enc.width, enc.height)
        enc.chunk_start_us = time.time_ns() // 1_000
        enc.chunk_frames   = 0

    def _make_track(self, pid: str, tid: str, width: int, height: int) -> _TrackEncoder:
        out_dir = _resolve_or_create_subdir(self._out_dir, pid)
        encoder = self._create_encoder(width, height)
        log.info("recorder  new track  pid=%r  dir=%s  track=%r  %dx%d  bitrate=%d",
                 pid, out_dir.name, tid, width, height, self._cfg.bitrate)
        return _TrackEncoder(
            encoder=encoder, out_dir=out_dir,
            width=width, height=height,
            chunk_start_us=time.time_ns() // 1_000,
        )

    def _create_encoder(self, width: int, height: int) -> object:
        return self._nvc.CreateEncoder(
            width, height, "NV12",
            True,   # usecpuinputbuffer
            gpu_id=self._cfg.gpu_id,
            codec="h264",
            preset="P4",
            tuning_info="high_quality",
            rc="vbr",
            fps=int(self._cfg.sample_fps),
            bitrate=self._cfg.bitrate,
            maxbitrate=self._cfg.bitrate,
            bf=0,
            repeat_sps_pps=1,
        )

    def _flush_chunk(self, enc: _TrackEncoder) -> None:
        if not enc.chunk_buf:
            return
        end_us    = time.time_ns() // 1_000
        data      = bytes(enc.chunk_buf)
        h264_path = enc.out_dir / f"{enc.chunk_start_us}.264"
        meta_path = enc.out_dir / f"{enc.chunk_start_us}.json"

        h264_path.write_bytes(data)
        meta_path.write_text(json.dumps({
            "start_us":   enc.chunk_start_us,
            "end_us":     end_us,
            "num_frames": enc.chunk_frames,
            "width":      enc.width,
            "height":     enc.height,
            "size_bytes": len(data),
        }))
        log.info("recorder  chunk  %s  %d frames  %d bytes",
                 h264_path.name, enc.chunk_frames, len(data))
        enc.chunk_buf.clear()
        self._prune_by_total_bytes()

    def _prune_by_total_bytes(self) -> None:
        """Keep total .264 chunk size under ``max_total_bytes`` across every
        participant directory. FIFO eviction by chunk start_us."""
        cap = self._cfg.max_total_bytes
        if cap <= 0:
            return
        with self._prune_lock:
            chunks: list[tuple[int, Path]] = []
            for pid_dir in self._out_dir.iterdir():
                if not pid_dir.is_dir():
                    continue
                for c in pid_dir.glob("*.264"):
                    try:
                        chunks.append((int(c.stem), c))
                    except ValueError:
                        continue
            chunks.sort()  # oldest first
            total = sum(c.stat().st_size for _, c in chunks)
            for _, c in chunks:
                if total <= cap:
                    break
                size = c.stat().st_size
                c.unlink(missing_ok=True)
                c.with_suffix(".json").unlink(missing_ok=True)
                total -= size

    # ── public API ────────────────────────────────────────────────────────────

    def close_participant(self, pid: str) -> None:
        with self._lock:
            keys = [k for k in self._encoders if k[0] == pid]
        for key in keys:
            enc = self._encoders.pop(key, None)
            if enc:
                with enc.lock:
                    try:
                        flushed = enc.encoder.EndEncode()
                        if flushed:
                            enc.chunk_buf.extend(flushed)
                    except Exception as e:
                        log.warning("recorder  flush error pid=%r: %s", pid, e)
                    self._flush_chunk(enc)


# ── pixel format conversion ───────────────────────────────────────────────────

def _to_nv12(data: bytes, width: int, height: int, fmt: object) -> np.ndarray | None:
    from xr_ai_agent import PixelFormat
    arr = np.frombuffer(data, dtype=np.uint8)

    if fmt == PixelFormat.NV12:
        return arr.reshape(height * 3 // 2, width)

    if fmt == PixelFormat.I420:
        y_size  = width * height
        uv_size = (width // 2) * (height // 2)
        Y = arr[:y_size]
        U = arr[y_size : y_size + uv_size]
        V = arr[y_size + uv_size :]
        uv = np.empty(uv_size * 2, dtype=np.uint8)
        uv[0::2] = U
        uv[1::2] = V
        return np.concatenate([Y, uv]).reshape(height * 3 // 2, width)

    if fmt == PixelFormat.RGB24:
        return _rgb24_to_nv12(arr.reshape(height, width, 3), width, height)

    if fmt == PixelFormat.RGBA:
        return _rgb24_to_nv12(arr.reshape(height, width, 4)[:, :, :3], width, height)

    if fmt == PixelFormat.BGRA:
        bgra = arr.reshape(height, width, 4)
        return _rgb24_to_nv12(bgra[:, :, [2, 1, 0]], width, height)

    log.warning("recorder  unsupported pixel format: %r", fmt)
    return None


def _rgb24_to_nv12(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)

    Y  = np.clip( 16.0 + 0.257 * R + 0.504 * G + 0.098 * B, 0, 255).astype(np.uint8)
    Cb = np.clip(128.0 - 0.148 * R - 0.291 * G + 0.439 * B, 0, 255).astype(np.uint8)
    Cr = np.clip(128.0 + 0.439 * R - 0.368 * G - 0.071 * B, 0, 255).astype(np.uint8)

    Cb2 = Cb[0::2, 0::2]
    Cr2 = Cr[0::2, 0::2]
    uv  = np.empty((height // 2) * width, dtype=np.uint8)
    uv[0::2] = Cb2.ravel()
    uv[1::2] = Cr2.ravel()
    return np.concatenate([Y.ravel(), uv]).reshape(height * 3 // 2, width)


def _safe_name(pid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)


def _resolve_or_create_subdir(root: Path, raw: str) -> Path:
    """Find or create the per-participant subdirectory for *raw*.

    Disambiguates collisions (two raw identities mapping to the same
    ``_safe_name``) by appending a counter suffix. Each subdir contains
    a ``.identity`` file holding the raw identity verbatim — downstream
    listing tools read it to recover the original name.
    """
    safe = _safe_name(raw)
    suffix = 1
    while True:
        name      = safe if suffix == 1 else f"{safe}_{suffix}"
        candidate = root / name
        sidecar   = candidate / ".identity"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            sidecar.write_text(raw, encoding="utf-8")
            return candidate
        if sidecar.exists() and sidecar.read_text(encoding="utf-8") == raw:
            return candidate
        # Pre-sidecar legacy dirs: if there's no .identity file and the
        # raw name already equals safe, claim it (and write the sidecar).
        if not sidecar.exists() and raw == safe and suffix == 1:
            sidecar.write_text(raw, encoding="utf-8")
            return candidate
        suffix += 1
