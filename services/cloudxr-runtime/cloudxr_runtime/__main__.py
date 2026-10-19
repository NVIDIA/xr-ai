# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cloudxr_runtime — run the CloudXR OpenXR runtime for XR AI.

Fetches the pinned CloudXR Runtime SDK on first use, starts the native
service (libcloudxr.so) in a subprocess, waits for it to become ready, then
runs the WSS proxy (port 48322) required by WebRTC device profiles. Native
device profiles skip the WSS step.

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_launcher import is_native_profile, read_device_profile
from xr_ai_logging import setup_logging

from . import _wss
from ._runtime import (
    Layout,
    check_eula,
    latest_runtime_log,
    prepare_run_dir,
    publish_env,
    start_service,
    stop_service,
)
from ._sdk import SDK_VERSION, SDKError, ensure_sdk


async def _wait_with_progress(
    runtime_proc: subprocess.Popen[bytes],
    stop: asyncio.Future,
    lock_file: Path,
    timeout_sec: float = 120.0,
) -> bool:
    """Poll for runtime_started lock file; exits early if stop is set or process dies."""
    deadline = asyncio.get_running_loop().time() + timeout_sec
    elapsed = 0
    while asyncio.get_running_loop().time() < deadline:
        if stop.done() or runtime_proc.poll() is not None:
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


def _xr_gpu_env(idx: int) -> dict[str, str] | None:
    """Return the env vars that pin Vulkan, CUDA, and Mesa to CUDA GPU *idx*.

    All three selectors are required: the CloudXR compositor runs on Vulkan
    and needs the matching CUDA device for interop, so on a multi-GPU host
    Vulkan and CUDA can otherwise land on different physical GPUs.

    Returns ``None`` (after logging a warning) when ``nvidia-smi`` is
    missing or fails, reports no GPUs, *idx* is not in the reported
    indices, or the PCI bus_id does not parse. Callers should treat
    ``None`` as "skip pinning" and continue startup.
    """
    if not shutil.which("nvidia-smi"):
        logger.warning("nvidia-smi not on PATH; skipping XR-side GPU pinning")
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5.0,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("nvidia-smi failed ({}); skipping XR-side GPU pinning", exc)
        return None

    by_index: dict[int, str] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            by_index[int(parts[0])] = parts[1]
        except ValueError:
            continue

    if not by_index:
        logger.warning("nvidia-smi reported no GPUs; skipping XR-side GPU pinning")
        return None

    bus_id = by_index.get(idx)
    if bus_id is None:
        logger.warning(
            "gpu_index={} not in nvidia-smi indices {}; skipping XR-side GPU pinning",
            idx, sorted(by_index.keys()),
        )
        return None

    # nvidia-smi reports bus_id as "00000000:XX:00.0" with an 8-hex domain.
    # Vulkan and Mesa selectors want the standard 4-hex-digit form.
    try:
        domain8, bus, dev_func = bus_id.split(":")
        dev, func = dev_func.split(".")
        domain = domain8[-4:]
    except ValueError:
        logger.warning(
            "could not parse PCI bus_id {!r}; skipping XR-side GPU pinning",
            bus_id,
        )
        return None

    return {
        "CUDA_VISIBLE_DEVICES":    str(idx),
        "VK_LOADER_DEVICE_SELECT": f"PCI:{bus}:{dev}:{func}",
        "DRI_PRIME":               f"pci-{domain}_{bus}_{dev}_{func}",
    }


def _append_env_to_file(path: Path, env: dict[str, str]) -> None:
    """Append ``export KEY=value`` lines to *path* so consumers that source
    the cloudxr env file inherit *env*."""
    lines = [f"export {k}={shlex.quote(v)}\n" for k, v in env.items()]
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(lines)


async def _run(
    cfg: dict,
    ready_file: Path | None = None,
    config_path: Path | None = None,
) -> None:
    layout   = Layout(Path(cfg.get("cloudxr_install_dir", "~/.cloudxr")).expanduser().absolute())
    env_file = cfg.get("cloudxr_env_config")

    # Environment variables such as NV_DEVICE_PROFILE are respected, with config values used as defaults.
    for key, val in cfg.get("cloudxr_env", {}).items():
        os.environ.setdefault(key, str(val))

    # XR-side GPU pinning. Set on os.environ before publish_env so the
    # process that hosts the native CloudXR service inherits the three selectors.
    pinning: dict[str, str] | None = None
    gpu_idx = cfg.get("gpu_index")
    if gpu_idx is None:
        logger.info("cloudxr: gpu_index unset; not pinning")
    else:
        pinning = _xr_gpu_env(int(gpu_idx))
        if pinning is not None:
            os.environ.update(pinning)
            logger.info(
                "cloudxr: pinning to GPU {} ({})",
                gpu_idx, pinning["VK_LOADER_DEVICE_SELECT"],
            )

    publish_env(layout, Path(env_file) if env_file else None)

    # Propagate pinning into cloudxr.env so the scene process inherits the same
    # GPU selectors when it spawns LOVR.
    if pinning is not None:
        _append_env_to_file(layout.env_file, pinning)

    try:
        sdk = ensure_sdk(layout.install_dir)
    except SDKError as exc:
        logger.error("{}", exc)
        sys.exit(1)
    check_eula(layout, sdk, accept=bool(cfg.get("accept_eula")))
    prepare_run_dir(layout, sdk)
    logs_dir = layout.logs_dir

    # Set up stop signal handling before anything async — so SIGTERM during startup
    # cleanly cancels the wait rather than being ignored or causing confusing ordering.
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def _on_signal() -> None:
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    runtime_proc = start_service(layout, sdk)

    logger.info("CloudXR {}", SDK_VERSION)
    logger.info("Waiting for CloudXR runtime…")

    try:
        ready = await _wait_with_progress(runtime_proc, stop, layout.ready_sentinel)
        if not ready:
            if stop.done():
                return  # cancelled by signal — clean exit
            native_log = latest_runtime_log(layout) or logs_dir
            rc = runtime_proc.poll()
            if rc is not None:
                logger.error(
                    "CloudXR runtime exited (rc={}) before signaling ready.\n"
                    "  Native log: {}\n"
                    "  Service stderr: {}",
                    rc, native_log, logs_dir / "runtime_stderr.log",
                )
                sys.exit(rc or 1)
            logger.error(
                "CloudXR runtime did not become ready within 120 s "
                "(process still alive — readiness lock file missing).\n"
                "  Native log: {}",
                native_log,
            )
            sys.exit(1)

        cxr_log = latest_runtime_log(layout) or logs_dir
        logger.info("CloudXR runtime:   ready  log: {}", cxr_log)
        logger.info("Activate CloudXR environment: source {}", layout.env_file)
        if ready_file:
            ready_file.touch()

        # Native-transport profiles connect directly; only WebRTC profiles need
        # the WSS signaling proxy.
        profile = read_device_profile(config_path)
        if is_native_profile(profile):
            logger.info("native device profile {}, skipping WSS proxy", profile)
            await stop
        else:
            try:
                await _wss.run(layout.install_dir / "certs", stop)
            except RuntimeError as exc:
                logger.error(
                    "CloudXR WSS proxy failed: {}\n"
                    "  If another process owns port 48322, stop it first:\n"
                    "    sudo fuser -k 48322/tcp",
                    exc,
                )
    finally:
        stop_service(runtime_proc)

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

    asyncio.run(_run(cfg, ready_file=our_ns.ready_file, config_path=our_ns.config))


if __name__ == "__main__":
    run()
