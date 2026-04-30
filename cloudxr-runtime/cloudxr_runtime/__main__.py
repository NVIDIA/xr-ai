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
            print(f"  [{elapsed}s] still waiting for CloudXR runtime…")
        await asyncio.sleep(1)
        elapsed += 1
    return False


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


async def _run(cfg: dict) -> None:
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
    print(f"Isaac Teleop {isaacteleop_version}  CloudXR {cxr_ver}")
    print("Waiting for CloudXR runtime…")

    try:
        ready = await _wait_with_progress(runtime_proc, stop)
        if not ready:
            if stop.done():
                return  # cancelled by signal — clean exit
            if not runtime_proc.is_alive() and runtime_proc.exitcode != 0:
                print(
                    f"CloudXR runtime exited (code {runtime_proc.exitcode}).\n"
                    "  Check for leftover containers: docker ps --filter name=cloudxr\n"
                    f"  Native log: {latest_runtime_log() or logs_dir}",
                    file=sys.stderr,
                )
                sys.exit(runtime_proc.exitcode or 1)
            print(
                "CloudXR runtime did not become ready within 120 s.\n"
                f"  Native log: {latest_runtime_log() or logs_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        cxr_log = latest_runtime_log() or logs_dir
        print(f"CloudXR runtime:   ready  log: {cxr_log}")
        print(f"Activate CloudXR environment: source {env_cfg.env_filepath()}")

        from datetime import datetime, timezone
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        wss_log = logs_dir / f"wss.{ts}.log"
        try:
            await wss_run(log_file_path=wss_log, stop_future=stop)
        except RuntimeError as exc:
            print(
                f"CloudXR WSS proxy failed: {exc}\n"
                "  If another process owns port 48322, stop it first:\n"
                "    sudo fuser -k 48322/tcp",
                file=sys.stderr,
            )
    finally:
        terminate_or_kill_runtime(runtime_proc)

    print("Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    our_ns, _ = p.parse_known_args()

    cfg: dict = {}
    if our_ns.config:
        cfg = _load_config(our_ns.config)

    asyncio.run(_run(cfg))


if __name__ == "__main__":
    run()
