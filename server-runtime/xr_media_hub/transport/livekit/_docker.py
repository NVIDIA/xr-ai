# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LiveKit server Docker lifecycle.

Runs the container as a foreground subprocess so Python owns the process
directly — no detach, no 'docker compose down' race.  Stopping is just
SIGTERM → wait → SIGKILL fallback, which is always reliable.

Failure surfaces
----------------
We surface the operator-fixable failure modes as :class:`StartupError`
so the entry point prints the banner and exits cleanly:

* ``docker`` not on PATH (``FileNotFoundError`` from ``exec``).
* ``docker run`` exits before the LiveKit port opens — its verbatim
  stdout/stderr is captured and embedded in the banner. Most common
  causes are the daemon not running, the user not in the ``docker``
  group, or an image-pull failure; the docker output names which.
* ``docker run`` is alive but the port never opens within the timeout
  — also surfaced as :class:`StartupError` with whatever output was
  captured up to that point.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import signal
import subprocess
import tempfile
from collections import deque
from pathlib import Path

from loguru import logger

from xr_media_hub._errors import StartupError

from .config import LiveKitConnectorConfig

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

_STOP_TIMEOUT = 10.0          # seconds to wait for graceful SIGTERM before SIGKILL
_READY_TIMEOUT = 30.0         # seconds to wait for the LiveKit port to open
_OUTPUT_CAPTURE_BYTES = 4096  # how much subprocess output to retain for diagnostics
_BANNER = "━" * 56

_CONTAINER_NAME = "xr-ai-livekit-server"


class LiveKitDocker:
    def __init__(self, cfg: LiveKitConnectorConfig) -> None:
        self._cfg    = cfg
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._proc:   asyncio.subprocess.Process | None  = None
        self._output: deque[bytes] = deque()
        self._output_size: int = 0
        self._log_task: asyncio.Task | None = None

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

        # Remove any stale container from a previous crashed run.
        subprocess.run(
            ["docker", "rm", "-f", _CONTAINER_NAME],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        logger.info("Starting LiveKit container (port {})…", self._cfg.lk_port_ws)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm",
                "--name", _CONTAINER_NAME,
                "--network", "host",
                "-v", f"{cfg_path}:/etc/livekit.yaml:ro",
                "livekit/livekit-server:latest",
                "--config", "/etc/livekit.yaml",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            self._cleanup_tmpdir()
            raise _docker_missing() from None

        # Belt-and-suspenders: kill the container if Python exits without stop().
        atexit.register(self._atexit_kill)

        self._log_task = asyncio.create_task(self._drain_logs(), name="livekit-logs")
        try:
            await self._wait_ready(self._cfg.lk_port_ws)
        except StartupError:
            # Make sure the subprocess is reaped and the tmpdir cleaned up
            # even though the lifecycle never completed.
            await self._teardown_failed_start()
            raise
        logger.info(
            "LiveKit container ready on port {}  pid={}",
            self._cfg.lk_port_ws, self._proc.pid,
        )

    async def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None

        if proc.returncode is not None:
            logger.info("LiveKit container already exited (rc={})", proc.returncode)
            await self._cancel_log_task()
            self._cleanup_tmpdir()
            return

        logger.info("Stopping LiveKit container (pid={})…", proc.pid)
        # docker stop sends SIGTERM to the container's PID 1 then waits;
        # this is reliable even if the docker run process is also killed
        # externally (e.g. by the launcher's killpg).
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["docker", "stop", "--time", str(int(_STOP_TIMEOUT)), _CONTAINER_NAME],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ),
            )
            logger.info("LiveKit container stopped")
        except Exception:
            logger.opt(exception=True).warning("docker stop failed — killing docker run directly")
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass

        await self._cancel_log_task()
        self._cleanup_tmpdir()

    # ── internals ─────────────────────────────────────────────────────────────

    async def _drain_logs(self) -> None:
        """
        Forward container output to the Python logger and retain the first
        ``_OUTPUT_CAPTURE_BYTES`` for diagnostics if startup fails.
        """
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            async for line in self._proc.stdout:
                if self._output_size < _OUTPUT_CAPTURE_BYTES:
                    self._output.append(line)
                    self._output_size += len(line)
                logger.debug("[livekit] {}", line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "LiveKit log drainer crashed — captured output may be truncated",
            )

    async def _wait_ready(self, port: int) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _READY_TIMEOUT
        while True:
            if self._proc is not None and self._proc.returncode is not None:
                # Give the log drainer a beat to flush any final output before
                # we read the captured buffer.
                await asyncio.sleep(0.1)
                raise _docker_exited_early(self._proc.returncode, self._captured())
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                if loop.time() >= deadline:
                    raise _docker_ready_timeout(_READY_TIMEOUT, port, self._captured())
                await asyncio.sleep(0.5)

    def _captured(self) -> str:
        if not self._output:
            return "(no output captured)"
        text = b"".join(self._output).decode("utf-8", errors="replace").rstrip()
        return text or "(empty output)"

    async def _teardown_failed_start(self) -> None:
        """Kill the subprocess (if still alive) and free the temp dir."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=_STOP_TIMEOUT)
            except asyncio.TimeoutError:
                pass
        await self._cancel_log_task()
        self._cleanup_tmpdir()

    async def _cancel_log_task(self) -> None:
        """Cancel and await the log drainer task so no output lines are leaked."""
        task = self._log_task
        self._log_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        # We just cancelled; CancelledError is the success path and any other
        # drainer error is moot now that the container is gone.
        except (asyncio.CancelledError, Exception):
            pass

    def _cleanup_tmpdir(self) -> None:
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _atexit_kill(self) -> None:
        """Last-resort synchronous stop registered with atexit."""
        if self._proc is None or self._proc.returncode is not None:
            return
        # docker stop is reliable even under SIGKILL to docker run itself.
        subprocess.run(
            ["docker", "stop", "--time", "5", _CONTAINER_NAME],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


# ── error builders ───────────────────────────────────────────────────────────


def _docker_missing() -> StartupError:
    lines = [
        "",
        _BANNER,
        "  Docker not found — refusing to start",
        _BANNER,
        "  The LiveKit transport runs inside a Docker container, but the",
        "  `docker` command is not on PATH.",
        "",
        "  Install Docker Engine and ensure your user can run containers:",
        "    https://docs.docker.com/engine/install/",
        _BANNER,
    ]
    return StartupError("\n".join(lines))


def _docker_exited_early(rc: int | None, output: str) -> StartupError:
    lines = [
        "",
        _BANNER,
        "  LiveKit container failed to start",
        _BANNER,
        f"  Exit code : {rc}",
        "",
        "  docker output (verbatim, first %d bytes):" % _OUTPUT_CAPTURE_BYTES,
        _indent(output),
        _BANNER,
    ]
    return StartupError("\n".join(lines))


def _docker_ready_timeout(timeout: float, port: int, output: str) -> StartupError:
    lines = [
        "",
        _BANNER,
        "  LiveKit container did not become ready in time",
        _BANNER,
        f"  Port    : {port}",
        f"  Timeout : {timeout:.0f}s",
        "",
        "  docker output so far (verbatim):",
        _indent(output),
        _BANNER,
    ]
    return StartupError("\n".join(lines))


def _indent(text: str, prefix: str = "    ") -> str:
    if not text:
        return f"{prefix}(no output)"
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())
