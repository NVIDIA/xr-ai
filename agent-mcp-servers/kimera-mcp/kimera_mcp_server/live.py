# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live (streaming) client for the Kimera-VIO C++ wrapper.

Spawns the ``kimera_vio`` docker image as a long-running container
that runs ``kimera_live_vio``.  The container listens on an AF_UNIX
socket inside ``/sock`` which we bind-mount from a host directory, so
this Python process can connect via a normal :class:`socket.socket`.

Protocol mirrors ``kimera_live_vio.cpp``: 12-byte little-endian header
``(msg_type, payload_len, reserved)`` followed by ``payload_len`` bytes.
Every request gets exactly one response.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import socket
import struct
import subprocess
import threading
import time

import numpy as np
from loguru import logger

from .build import ensure_image


# Wire-format type IDs (must match kimera_live_vio.cpp).
_MSG_PING       = 1
_MSG_INTRINSICS = 2
_MSG_IMU        = 3
_MSG_FRAME      = 4
_MSG_RESET      = 5

_RSP_PONG  = 11
_RSP_OK    = 12
_RSP_POSE  = 13
_RSP_ERROR = 20

_HDR = struct.Struct("<III")   # msg_type, payload_len, reserved


@dataclasses.dataclass(frozen=True)
class LivePose:
    have_pose:    bool
    state:        str                 # "ok" | "uninitialized" | "lost"
    ts_ns:        int
    translation:  tuple[float, float, float]
    quaternion:   tuple[float, float, float, float]   # (w, x, y, z)


class KimeraLiveError(RuntimeError):
    """Raised on socket / protocol errors talking to ``kimera_live_vio``."""


class KimeraLiveClient:
    """Manages the ``kimera_vio`` container + a socket connection to its
    streaming binary.  Thread-safe: a mutex serialises all requests so
    multiple FastMCP coroutines can share one client."""

    def __init__(
        self, *,
        sock_dir:        pathlib.Path,
        docker_image:    str  = "kimera_vio",
        deps_image:      str  = "kimera_vio_deps",
        container_name:  str  = "xr_ai_kimera_live",
        params_folder_in_image: str = "/opt/kimera-params/EurocMonoLive",
        startup_timeout_s: float = 30.0,
        build_if_missing:   bool = True,
        kimera_vio_repo:    str  = "https://github.com/MIT-SPARK/Kimera-VIO.git",
        kimera_vio_src_cache: pathlib.Path = pathlib.Path("~/.cache/xr-ai/Kimera-VIO"),
    ) -> None:
        self._sock_dir   = pathlib.Path(sock_dir).expanduser()
        self._sock_path  = self._sock_dir / "kimera-vio.sock"
        self._image      = docker_image
        self._deps_image = deps_image
        self._container  = container_name
        self._params     = params_folder_in_image
        self._timeout    = float(startup_timeout_s)
        self._build_if_missing     = bool(build_if_missing)
        self._kimera_vio_repo      = kimera_vio_repo
        self._kimera_vio_src_cache = pathlib.Path(kimera_vio_src_cache).expanduser()

        self._sock:    socket.socket | None = None
        self._lock     = threading.Lock()
        self._started  = False
        # Remember the last intrinsics so an auto-reconnect after a
        # crash can re-apply them without depending on a caller refresh.
        self._last_intrinsics: tuple | None = None
        # Reentry guard — prevents the auto-reconnect path from
        # itself triggering another reconnect on the intrinsics
        # re-apply if the new container is still flaky.
        self._in_reconnect = False

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the docker container (idempotent) and connect to its
        socket.  Blocks until the binary is reachable or
        ``startup_timeout_s`` elapses.  Builds the docker image on the
        fly if ``build_if_missing`` is enabled and ``docker_image``
        is not already present on the host."""
        if self._started:
            return
        if self._build_if_missing:
            ensure_image(
                image_tag=self._image,
                deps_tag =self._deps_image,
                src_cache=self._kimera_vio_src_cache,
                repo_url =self._kimera_vio_repo,
            )
        self._sock_dir.mkdir(parents=True, exist_ok=True)
        # Clean up any stale socket from a prior run.
        try:
            self._sock_path.unlink()
        except FileNotFoundError:
            pass
        # If the named container is still around from a previous crash,
        # capture its logs *before* removing it so we can see why it
        # died.  We drop --rm so the next crash leaves the container
        # around for the same reason.
        self._dump_logs()
        subprocess.run(
            ["docker", "rm", "-f", self._container],
            check=False, capture_output=True,
        )
        cmd = [
            "docker", "run", "-d",
            "--name", self._container,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{self._sock_dir}:/sock:rw",
            self._image,
            "--socket_path=/sock/kimera-vio.sock",
            f"--params_folder_path={self._params}",
        ]
        logger.info("[kimera-live] spawning container: {}", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except subprocess.CalledProcessError as exc:
            raise KimeraLiveError(
                f"docker run failed (rc={exc.returncode}): "
                f"{exc.stderr.decode(errors='replace')[:500]}"
            ) from exc

        # Wait for the binary to bind the socket.
        t_deadline = time.monotonic() + self._timeout
        while time.monotonic() < t_deadline:
            if self._sock_path.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(str(self._sock_path))
                    self._sock = s
                    break
                except (FileNotFoundError, ConnectionRefusedError, OSError):
                    pass
            time.sleep(0.1)

        if self._sock is None:
            self._dump_logs()
            self.stop()
            raise KimeraLiveError(
                f"kimera_live_vio did not open {self._sock_path} within "
                f"{self._timeout:.0f}s — check container logs above"
            )
        # Quick handshake — confirms the protocol is responsive.
        self._roundtrip(_MSG_PING, b"")
        self._started = True
        logger.info("[kimera-live] connected to {}", self._sock_path)

    def stop(self) -> None:
        with self._lock:
            if self._sock is not None:
                try: self._sock.close()
                except Exception: pass
                self._sock = None
        self._started = False
        # Capture the container's stderr in case it died for a reason
        # we want to debug after shutdown.  Best-effort — never raises.
        self._dump_logs()
        subprocess.run(
            ["docker", "rm", "-f", self._container],
            check=False, capture_output=True,
        )

    # ── tool API ───────────────────────────────────────────────────────

    def set_intrinsics(
        self, *, width: int, height: int,
        fx: float, fy: float, cx: float, cy: float,
        k1: float = 0.0, k2: float = 0.0,
        p1: float = 0.0, p2: float = 0.0,
    ) -> None:
        # Cache so a crash-recovery reconnect can re-apply them without
        # waiting for the caller to send another camera_meta.
        self._last_intrinsics = (
            int(width), int(height),
            float(fx), float(fy), float(cx), float(cy),
            float(k1), float(k2), float(p1), float(p2),
        )
        payload = struct.pack("<II8d", *self._last_intrinsics)
        rt, body = self._roundtrip(_MSG_INTRINSICS, payload)
        if rt != _RSP_OK:
            raise KimeraLiveError(self._decode_err(rt, body, "INTRINSICS"))

    def push_imu(self, samples: list[tuple[int, tuple[float, ...]]]) -> int:
        """`samples` is a list of (ts_ns, (gx, gy, gz, ax, ay, az)).
        Returns the number of samples sent."""
        if not samples:
            return 0
        # Header: u32 count, then per-sample (u64 ts_ns, 6 × f64).
        buf = bytearray(struct.pack("<I", len(samples)))
        for ts_ns, v in samples:
            if len(v) != 6:
                continue
            buf += struct.pack("<Q6d", int(ts_ns), *map(float, v))
        rt, body = self._roundtrip(_MSG_IMU, bytes(buf))
        if rt != _RSP_OK:
            raise KimeraLiveError(self._decode_err(rt, body, "IMU"))
        return len(samples)

    def send_frame(self, *, ts_ns: int, gray: np.ndarray) -> LivePose:
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("send_frame expects a 2-D uint8 grayscale array")
        h, w = gray.shape
        buf = struct.pack("<QII", int(ts_ns), int(w), int(h)) + gray.tobytes()
        rt, body = self._roundtrip(_MSG_FRAME, buf)
        if rt != _RSP_POSE:
            raise KimeraLiveError(self._decode_err(rt, body, "FRAME"))
        # Body: u64 ts_ns, 7 × f64, u32 state.
        ts, tx, ty, tz, qw, qx, qy, qz, state = struct.unpack("<Q7dI", body)
        state_name = {0: "ok", 1: "uninitialized", 2: "lost"}.get(state, "lost")
        # Renormalise q so consumers downstream don't have to.
        n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
        if n > 1e-9:
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        return LivePose(
            have_pose   =(state == 0),
            state       =state_name,
            ts_ns       =int(ts),
            translation =(tx, ty, tz),
            quaternion  =(qw, qx, qy, qz),
        )

    def reset(self) -> None:
        rt, body = self._roundtrip(_MSG_RESET, b"")
        if rt != _RSP_OK:
            raise KimeraLiveError(self._decode_err(rt, body, "RESET"))

    # ── low-level transport ────────────────────────────────────────────

    def _roundtrip(self, msg_type: int, payload: bytes) -> tuple[int, bytes]:
        """Send a request and read its response.  If the socket has
        died (because ``kimera_live_vio`` crashed mid-stream), we
        respawn the container, re-apply the last intrinsics, and retry
        once.  The retry only fires for non-handshake messages, so a
        broken pipe during PING surfaces as a clean error instead of
        looping forever."""
        for attempt in (0, 1):
            with self._lock:
                if self._sock is None and attempt > 0:
                    # First attempt already failed; the previous block
                    # cleared _sock and tried to reconnect.  Don't retry
                    # again from inside the lock.
                    raise KimeraLiveError("not connected")
                if self._sock is not None:
                    hdr = _HDR.pack(msg_type, len(payload), 0)
                    try:
                        self._sock.sendall(hdr + payload)
                        rsp_hdr = self._recv_exact(_HDR.size)
                        r_type, r_len, _ = _HDR.unpack(rsp_hdr)
                        body = self._recv_exact(r_len) if r_len > 0 else b""
                        return r_type, body
                    except (OSError, KimeraLiveError) as exc:
                        # Mark connection dead but keep the lock so the
                        # outer caller doesn't race with a recover.
                        try: self._sock.close()
                        except Exception: pass
                        self._sock    = None
                        self._started = False
                        last_exc      = exc
                else:
                    last_exc = KimeraLiveError("not connected")

            # Drop the lock before doing the recovery, which itself
            # acquires the lock when it reconnects.
            if (attempt == 0
                    and msg_type != _MSG_PING
                    and not self._in_reconnect):
                logger.warning(
                    "[kimera-live] socket dead ({}), reconnecting…", last_exc,
                )
                try:
                    self._reconnect()
                except Exception as exc:
                    raise KimeraLiveError(
                        f"reconnect after socket error failed: {exc}",
                    ) from exc
                continue
            raise KimeraLiveError(f"socket error: {last_exc}") from last_exc

        # Unreachable — both loop iterations either return or raise.
        raise KimeraLiveError("unreachable")

    def _reconnect(self) -> None:
        """Tear down the dead container, spawn a new one, re-apply the
        cached intrinsics so the next FRAME has a working pipeline.

        Reentry-guarded: inner calls to _roundtrip() during this
        method won't themselves trigger another reconnect, so a
        chronically-crashing binary surfaces as an error to the caller
        instead of an infinite loop."""
        self._in_reconnect = True
        try:
            self._started = False
            self.start()
            if self._last_intrinsics is not None:
                try:
                    payload = struct.pack("<II8d", *self._last_intrinsics)
                    rt, body = self._roundtrip(_MSG_INTRINSICS, payload)
                    if rt != _RSP_OK:
                        logger.warning(
                            "[kimera-live] failed to re-apply intrinsics "
                            "after reconnect: {}",
                            self._decode_err(rt, body, "INTRINSICS"),
                        )
                except KimeraLiveError as exc:
                    logger.warning(
                        "[kimera-live] re-apply intrinsics failed: {}", exc,
                    )
        finally:
            self._in_reconnect = False

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise KimeraLiveError("socket closed by peer")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode_err(rt: int, body: bytes, ctx: str) -> str:
        if rt == _RSP_ERROR and len(body) >= 4:
            (n,) = struct.unpack("<I", body[:4])
            return f"{ctx}: {body[4:4+n].decode(errors='replace')}"
        return f"{ctx}: unexpected response type {rt}"

    def _dump_logs(self) -> None:
        out = subprocess.run(
            ["docker", "logs", self._container],
            check=False, capture_output=True, timeout=5,
        )
        if out.stdout:
            logger.warning("[kimera-live] stdout:\n{}", out.stdout.decode(errors="replace"))
        if out.stderr:
            logger.warning("[kimera-live] stderr:\n{}", out.stderr.decode(errors="replace"))
