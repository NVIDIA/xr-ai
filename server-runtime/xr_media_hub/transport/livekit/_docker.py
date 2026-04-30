"""
LiveKit server Docker lifecycle.

Runs the container as a foreground subprocess so Python owns the process
directly — no detach, no 'docker compose down' race.  Stopping is just
SIGTERM → wait → SIGKILL fallback, which is always reliable.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
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

_STOP_TIMEOUT = 10.0   # seconds to wait for graceful SIGTERM before SIGKILL


class LiveKitDocker:
    def __init__(self, cfg: LiveKitConnectorConfig) -> None:
        self._cfg    = cfg
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._proc:   asyncio.subprocess.Process | None  = None

    async def start(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="xr_livekit_")
        cfg_path = Path(self._tmpdir.name) / "livekit.yaml"
        cfg_path.write_text(_LIVEKIT_YAML.format(
            lk_port_ws  = self._cfg.lk_port_ws,
            lk_port_tcp = self._cfg.lk_port_tcp,
            lk_port_udp = self._cfg.lk_port_udp,
            api_key     = self._cfg.api_key,
            api_secret  = self._cfg.api_secret,
        ))

        log.info("Starting LiveKit container (port %d)…", self._cfg.lk_port_ws)
        self._proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm",
            "--network", "host",
            "-v", f"{cfg_path}:/etc/livekit.yaml:ro",
            "livekit/livekit-server:latest",
            "--config", "/etc/livekit.yaml",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Belt-and-suspenders: kill the container if Python exits without stop().
        atexit.register(self._atexit_kill)

        asyncio.create_task(self._drain_logs(), name="livekit-logs")
        await self._wait_ready(self._cfg.lk_port_ws)
        log.info("LiveKit container ready on port %d  pid=%d",
                 self._cfg.lk_port_ws, self._proc.pid)

    async def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None

        if proc.returncode is not None:
            log.info("LiveKit container already exited (rc=%d)", proc.returncode)
            self._cleanup_tmpdir()
            return

        log.info("Stopping LiveKit container (pid=%d)…", proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone

        try:
            await asyncio.wait_for(proc.wait(), timeout=_STOP_TIMEOUT)
            log.info("LiveKit container stopped (rc=%d)", proc.returncode)
        except asyncio.TimeoutError:
            log.warning("SIGTERM timed out after %.0fs — sending SIGKILL", _STOP_TIMEOUT)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            log.info("LiveKit container killed")

        self._cleanup_tmpdir()

    # ── internals ─────────────────────────────────────────────────────────────

    async def _drain_logs(self) -> None:
        """Forward container stdout/stderr to the Python logger."""
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            async for line in self._proc.stdout:
                log.debug("[livekit] %s", line.decode(errors="replace").rstrip())
        except Exception:
            pass

    async def _wait_ready(self, port: int, timeout: float = 30.0) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            if self._proc is not None and self._proc.returncode is not None:
                raise RuntimeError(
                    f"LiveKit container exited early (rc={self._proc.returncode})"
                )
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

    def _cleanup_tmpdir(self) -> None:
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _atexit_kill(self) -> None:
        """Last-resort synchronous kill registered with atexit."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
