"""
NVENC video recorder for XR-Media-Hub.

Records incoming video frames per participant to H.264 Annex B chunk files.
Each chunk starts with SPS+PPS+IDR and is independently decodable.

On-disk layout
--------------
    <out_dir>/<participant_id>/
        <start_us>.264    — raw H.264 Annex B, starts with IDR
        <start_us>.json   — chunk metadata sidecar

Sidecar JSON keys:
    start_us    int   chunk start time (Unix µs, same as filename stem)
    end_us      int   chunk end time (Unix µs, written when chunk is closed)
    num_frames  int   encoded frames in this chunk
    width       int   frame width in pixels
    height      int   frame height in pixels
    size_bytes  int   .264 file size

The video MCP server reads this directory directly — the hub exposes no HTTP API
for video.
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


@dataclass
class VideoRecorderConfig:
    out_dir:      str   = "/tmp/xr_recordings"
    chunk_frames: int   = 30        # frames per chunk (1 s at 30 fps)
    max_chunks:   int   = 300       # max chunks retained per participant (0 = unlimited)
    sample_fps:   float = 30.0
    bitrate:      int   = 4_000_000
    gpu_id:       int   = 0


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
        self._encoders:    dict[tuple[str, str], _TrackEncoder] = {}
        self._lock         = threading.Lock()
        self._min_interval = 1.0 / max(cfg.sample_fps, 1.0)

    # ── hub callback ──────────────────────────────────────────────────────────

    async def on_frame(self, view: "SlotView") -> None:
        import asyncio
        sig  = view.signal
        data = bytes(view.data)
        await asyncio.get_running_loop().run_in_executor(
            None, self._encode_frame,
            sig.participant_id, sig.track_id,
            data, sig.width, sig.height, sig.fmt,
        )

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
                self._flush_chunk(enc)
                enc.encoder        = self._create_encoder(width, height)
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
        out_dir = self._out_dir / _safe_name(pid)
        out_dir.mkdir(parents=True, exist_ok=True)
        encoder = self._create_encoder(width, height)
        log.info("recorder  new track  pid=%r  track=%r  %dx%d  bitrate=%d",
                 pid, tid, width, height, self._cfg.bitrate)
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
        self._prune(enc.out_dir)

    def _prune(self, out_dir: Path) -> None:
        if not self._cfg.max_chunks:
            return
        chunks = sorted(out_dir.glob("*.264"))
        for old in chunks[: max(0, len(chunks) - self._cfg.max_chunks)]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

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
