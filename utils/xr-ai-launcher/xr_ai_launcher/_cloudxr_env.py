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

__all__ = ["XR_RUNTIME_VAR", "load_cloudxr_env"]


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
    log.info("cloudxr env sourced from %s  (%s=%s)",
             path, XR_RUNTIME_VAR, os.environ.get(XR_RUNTIME_VAR, "<missing>"))
