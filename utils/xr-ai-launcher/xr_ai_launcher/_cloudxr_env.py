# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CloudXR environment helper — source ``cloudxr.env`` into ``os.environ``.

With serial process startup (``run_stack`` launches cloudxr-runtime first and
waits for its ready file before starting any OpenXR consumer), callers no
longer need to poll for the env file — it already exists by the time they
start.  ``load_cloudxr_env`` is the only function needed.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger("xr_ai_launcher.cloudxr_env")

_EXPORT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# OpenXR runtime selector written by cloudxr-runtime into cloudxr.env.
XR_RUNTIME_VAR = "XR_RUNTIME_JSON"

# Profiles on CloudXR's direct native transport (skip the WSS proxy).
NATIVE_DEVICE_PROFILES = frozenset({"auto-native", "apple-vision-pro", "ipad-pro"})

_DEVICE_PROFILE_RE = re.compile(
    r"^\s*NV_DEVICE_PROFILE\s*:\s*[\"']?([\w-]+)[\"']?", re.MULTILINE
)

__all__ = [
    "XR_RUNTIME_VAR",
    "load_cloudxr_env",
    "NATIVE_DEVICE_PROFILES",
    "is_native_profile",
    "read_device_profile",
]


def is_native_profile(profile: str) -> bool:
    """True if *profile* names a native-transport CloudXR device profile."""
    return (profile or "").strip().lower() in NATIVE_DEVICE_PROFILES


def read_device_profile(yaml_path) -> str:
    """Return NV_DEVICE_PROFILE from the environment, or from *yaml_path* when unset."""
    env_val = os.environ.get("NV_DEVICE_PROFILE")
    if env_val:
        return env_val
    if not yaml_path:
        return ""
    try:
        with open(yaml_path) as f:
            text = f.read()
    except OSError:
        return ""
    m = _DEVICE_PROFILE_RE.search(text)
    return m.group(1) if m else ""


def load_cloudxr_env(path: Path) -> None:
    """Parse an ``export KEY=VALUE`` env file and merge into ``os.environ``."""
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
    log.debug("cloudxr env sourced from %s  (%s=%s)",
              path, XR_RUNTIME_VAR, os.environ.get(XR_RUNTIME_VAR, "<missing>"))
