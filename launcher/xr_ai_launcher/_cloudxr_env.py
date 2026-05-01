# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CloudXR environment helpers — wait for cloudxr-runtime's ``cloudxr.env`` file,
source it into ``os.environ``, and wait for its ``runtime_started`` lock file
to appear. Used by render-mcp and oxr-mcp before opening their OpenXR sessions.

Stdlib-only.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("xr_ai_launcher.cloudxr_env")

# Permissive identifier — matches the all-upper-case keys cloudxr-runtime
# emits today and any future mixed/lower-case ones. Note that downstream
# ``os.environ.get`` lookups remain case-sensitive on Linux, so a real
# lower-case variant would still need its own fix.
_EXPORT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# OpenXR runtime selector. Without this in os.environ an OpenXR client falls
# back to whatever system runtime is registered (Monado / nothing / a stale
# install) and renders into a swapchain CloudXR has no idea about.
XR_RUNTIME_VAR = "XR_RUNTIME_JSON"

__all__ = [
    "XR_RUNTIME_VAR",
    "load_cloudxr_env",
    "wait_for_cloudxr_env",
    "wait_for_cloudxr_runtime_started",
]


def load_cloudxr_env(path: Path) -> None:
    """Parse an export KEY=VALUE env file and merge into ``os.environ``."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _EXPORT_RE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            os.environ[key] = val
    log.info("cloudxr env sourced from %s  (%s=%s)",
             path, XR_RUNTIME_VAR, os.environ.get(XR_RUNTIME_VAR, "<missing>"))


async def wait_for_cloudxr_env(
    path: Path, *, timeout_sec: float = 60.0, log_prefix: str = "cloudxr-env",
) -> bool:
    """Wait until *path* exists and contains ``XR_RUNTIME_JSON``.

    Returns True on success, False if the timeout elapsed first.
    """
    deadline = asyncio.get_running_loop().time() + timeout_sec
    waited = 0
    while asyncio.get_running_loop().time() < deadline:
        if path.exists():
            try:
                text = path.read_text()
            except OSError:
                text = ""
            if XR_RUNTIME_VAR in text:
                return True
        if waited and waited % 5 == 0:
            log.info("%s: still waiting for %s (%ds)", log_prefix, path, waited)
        await asyncio.sleep(1)
        waited += 1
    return False


async def wait_for_cloudxr_runtime_started(
    *, timeout_sec: float = 120.0, log_prefix: str = "cloudxr-env",
) -> bool:
    """Poll for cloudxr's ``runtime_started`` lock file.

    Connecting before this lock appears leaves the OpenXR client with a
    half-built session and CloudXR streaming empty frames forever.

    Reads ``$NV_CXR_RUNTIME_DIR`` (set by ``load_cloudxr_env``). Returns
    True on success or False on timeout; if the env var is missing, logs
    a warning and returns True so the caller can decide to proceed.
    """
    run_dir = os.environ.get("NV_CXR_RUNTIME_DIR")
    if not run_dir:
        log.warning("%s: NV_CXR_RUNTIME_DIR not set; skipping runtime-ready wait",
                    log_prefix)
        return True
    lock = Path(run_dir) / "runtime_started"
    deadline = asyncio.get_running_loop().time() + timeout_sec
    waited = 0
    while asyncio.get_running_loop().time() < deadline:
        if lock.exists():
            return True
        if waited and waited % 5 == 0:
            log.info("%s: still waiting for cloudxr runtime ready %s (%ds)",
                     log_prefix, lock, waited)
        await asyncio.sleep(1)
        waited += 1
    return False
