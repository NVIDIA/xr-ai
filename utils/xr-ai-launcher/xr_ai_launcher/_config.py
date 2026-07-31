# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal config reads for the stdlib-only launcher."""

import re
from pathlib import Path


def read_config_scalar(path: Path, key: str, default: str = "") -> str:
    """Read one top-level YAML scalar without adding a YAML dependency."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(
        rf"^\s*{re.escape(key)}\s*:\s*(?:\"([^\"]*)\"|'([^']*)'|([^#\s]+))",
        text,
        re.MULTILINE,
    )
    if not match:
        return default
    return next(value for value in match.groups() if value is not None)
