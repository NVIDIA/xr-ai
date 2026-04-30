# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LiveKit server Docker lifecycle.

Writes a minimal livekit.yaml and docker-compose.yml to a temp directory,
starts the container, waits for the signaling port to accept connections, and
tears down on stop().
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from .config import LiveKitConnectorConfig

log = logging.getLogger(__name__)

_LIVEKIT_YAML = """\
port: {lk_port_ws}
rtc:
  tcp_port: {lk_port_tcp}
  udp_port: {lk_port_udp}
  use_external_ip: false
keys:
  {api_key}: {api_secret}
logging:
  json: false
  level: info
room:
  auto_create: true
"""

_COMPOSE = """\
services:
  livekit:
    image: livekit/livekit-server:latest
    command: --config /etc/livekit.yaml
    restart: "no"
    network_mode: host
    volumes:
      - {cfg_path}:/etc/livekit.yaml
"""


class LiveKitDocker:
    def __init__(self, cfg: LiveKitConnectorConfig) -> None:
        self._cfg = cfg
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._compose_path: str | None = None
        self._container_id: str | None = None

    async def start(self) -> None:
        """Write config files, start the container, wait for readiness."""
        self._tmpdir = tempfile.TemporaryDirectory(prefix="xr_livekit_")
        tmp = Path(self._tmpdir.name)

        cfg_text = _LIVEKIT_YAML.format(
            lk_port_ws=self._cfg.lk_port_ws,
            lk_port_tcp=self._cfg.lk_port_tcp,
            lk_port_udp=self._cfg.lk_port_udp,
            api_key=self._cfg.api_key,
            api_secret=self._cfg.api_secret,
        )
        cfg_path = tmp / "livekit.yaml"
        cfg_path.write_text(cfg_text)

        compose_text = _COMPOSE.format(cfg_path=str(cfg_path))
        compose_path = tmp / "docker-compose.yml"
        compose_path.write_text(compose_text)
        self._compose_path = str(compose_path)

        log.info("Starting LiveKit container (port %d)…", self._cfg.lk_port_ws)
        proc = await asyncio.create_subprocess_exec(
            "docker", "compose", "-f", self._compose_path,
            "up", "-d", "--pull", "missing",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker compose up failed:\n{stderr.decode()}")

        await self._wait_ready(self._cfg.lk_port_ws)

        # Record container ID for fallback stop.
        id_proc = await asyncio.create_subprocess_exec(
            "docker", "compose", "-f", self._compose_path, "ps", "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        id_out, _ = await id_proc.communicate()
        self._container_id = id_out.decode().strip().splitlines()[0] if id_out.strip() else None

        log.info("LiveKit container ready on port %d  id=%s",
                 self._cfg.lk_port_ws, self._container_id or "unknown")

    async def stop(self) -> None:
        if self._compose_path:
            log.info("Stopping LiveKit container…")
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose", "-f", self._compose_path, "down",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.warning(
                    "docker compose down exited %d — trying docker stop fallback\n%s",
                    proc.returncode,
                    (stderr or stdout or b"").decode(errors="replace").strip(),
                )
                await self._stop_by_id()
            else:
                log.info("LiveKit container stopped")
            self._compose_path = None
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None

    async def _stop_by_id(self) -> None:
        """Fallback: stop and remove the container by ID when compose down fails."""
        if not self._container_id:
            return
        for cmd in (["docker", "stop", self._container_id],
                    ["docker", "rm",   self._container_id]):
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.warning(
                    "%s failed (rc=%d): %s",
                    " ".join(cmd), proc.returncode,
                    stderr.decode(errors="replace").strip(),
                )
        self._container_id = None

    async def _wait_ready(self, port: int, timeout: float = 30.0) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"LiveKit port {port} not ready after {timeout}s"
                    )
                await asyncio.sleep(0.5)
