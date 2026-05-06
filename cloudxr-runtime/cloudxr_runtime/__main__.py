# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cloudxr_runtime — launcher for isaacteleop.cloudxr.

Starts the native CloudXR service (libcloudxr.so) in a subprocess, waits for
it to become ready, then runs isaacteleop's WSS proxy (port 48322) required by
the auto-webrtc device profile.  auto-native skips the WSS step.

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).
"""
from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import os
import signal
import sys
from pathlib import Path

import yaml
from isaacteleop import __version__ as isaacteleop_version
from isaacteleop.cloudxr.env_config import EnvConfig
from isaacteleop.cloudxr.runtime import (
    check_eula,
    latest_runtime_log,
    run as runtime_run,
    runtime_version,
    terminate_or_kill_runtime,
    wait_for_runtime_ready,
)
from isaacteleop.cloudxr.wss import run as wss_run
from loguru import logger
from xr_ai_logging import setup_logging


async def _wait_with_progress(
    runtime_proc: multiprocessing.Process,
    stop: asyncio.Future,
    timeout_sec: float = 120.0,
) -> bool:
    """Poll for runtime_started lock file; exits early if stop is set or process dies."""
    from isaacteleop.cloudxr.env_config import get_env_config
    lock_file = Path(get_env_config().openxr_run_dir()) / "runtime_started"
    deadline = asyncio.get_running_loop().time() + timeout_sec
    elapsed = 0
    while asyncio.get_running_loop().time() < deadline:
        if stop.done() or not runtime_proc.is_alive():
            return False
        if lock_file.exists():
            return True
        if elapsed > 0 and elapsed % 10 == 0:
            logger.debug("[{}s] still waiting for CloudXR runtime…", elapsed)
        await asyncio.sleep(1)
        elapsed += 1
    return False


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


async def _run(cfg: dict, ready_file: Path | None = None) -> None:
    install_dir = str(Path(cfg.get("cloudxr_install_dir", "~/.cloudxr")).expanduser())
    env_file    = cfg.get("cloudxr_env_config")

    for key, val in cfg.get("cloudxr_env", {}).items():
        os.environ[key] = str(val)

    env_cfg = EnvConfig.from_args(install_dir, env_file)
    check_eula(accept_eula=cfg.get("accept_eula") or None)
    logs_dir = env_cfg.ensure_logs_dir()

    # Set up stop signal handling before anything async — so SIGTERM during startup
    # cleanly cancels the wait rather than being ignored or causing confusing ordering.
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def _on_signal() -> None:
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    runtime_proc = multiprocessing.Process(target=runtime_run)
    runtime_proc.start()

    cxr_ver = runtime_version()
    logger.info("Isaac Teleop {}  CloudXR {}", isaacteleop_version, cxr_ver)
    logger.info("Waiting for CloudXR runtime…")

    try:
        ready = await _wait_with_progress(runtime_proc, stop)
        if not ready:
            if stop.done():
                return  # cancelled by signal — clean exit
            native_log = latest_runtime_log() or logs_dir
            if not runtime_proc.is_alive():
                # Process exited before signaling ready. exitcode may be None
                # for a tick after termination — treat that as a failure too.
                rc = runtime_proc.exitcode
                logger.error(
                    "CloudXR runtime exited (rc={}) before signaling ready.\n"
                    "  Native log: {}\n"
                    "  Leftover containers: docker ps --filter name=cloudxr",
                    rc if rc is not None else "?", native_log,
                )
                sys.exit(rc or 1)
            logger.error(
                "CloudXR runtime did not become ready within 120 s "
                "(process still alive — readiness lock file missing).\n"
                "  Native log: {}",
                native_log,
            )
            sys.exit(1)

        cxr_log = latest_runtime_log() or logs_dir
        logger.info("CloudXR runtime:   ready  log: {}", cxr_log)
        logger.info("Activate CloudXR environment: source {}", env_cfg.env_filepath())
        if ready_file:
            ready_file.touch()

        from datetime import datetime, timezone
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        wss_log = logs_dir / f"wss.{ts}.log"
        try:
            await wss_run(log_file_path=wss_log, stop_future=stop)
        except RuntimeError as exc:
            logger.error(
                "CloudXR WSS proxy failed: {}\n"
                "  If another process owns port 48322, stop it first:\n"
                "    sudo fuser -k 48322/tcp",
                exc,
            )
    finally:
        terminate_or_kill_runtime(runtime_proc)

    logger.info("Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("cloudxr")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    our_ns, _ = p.parse_known_args()

    cfg: dict = {}
    if our_ns.config:
        cfg = _load_config(our_ns.config)

    asyncio.run(_run(cfg, ready_file=our_ns.ready_file))


if __name__ == "__main__":
    run()
