# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal scalar reads for stdlib-only orchestrators."""
from __future__ import annotations

import re
from pathlib import Path


def read_config_scalar(path: Path, key: str, default: str = "") -> str:
    """Read one scalar from a top-level YAML worker configuration."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(
        rf"^[ \t]*{re.escape(key)}[ \t]*:[ \t]*"
        rf"(?:\"([^\"]*)\"|'([^']*)'|([^#\s]+))",
        text,
        re.MULTILINE,
    )
    if not match:
        return default
    return next(value for value in match.groups() if value is not None)
