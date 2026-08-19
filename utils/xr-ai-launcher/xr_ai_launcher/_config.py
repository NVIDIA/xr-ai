# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal scalar reads for stdlib-only orchestrators."""
from __future__ import annotations

import re
from pathlib import Path


def read_config_scalar(path: Path, key: str, default: str = "") -> str:
    """Read one scalar from a top-level YAML worker configuration."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return default

    key_pattern = re.compile(rf"^{re.escape(key)}[ \t]*:[ \t]*(.*)$")
    for line in lines:
        match = key_pattern.fullmatch(line)
        if match is None:
            continue
        raw = match.group(1).strip()
        if not raw or raw[0] in "#|>":
            return default
        if raw[0] in "\"'":
            quote = raw[0]
            end = raw.find(quote, 1)
            suffix = raw[end + 1:].strip() if end >= 0 else ""
            if end < 0 or (suffix and not suffix.startswith("#")):
                return default
            return raw[1:end] or default
        comment = re.search(r"[ \t]+#", raw)
        return raw[:comment.start()].rstrip() if comment else raw
    return default


def read_service_port(path: Path) -> int | None:
    """Read a service's top-level HTTP port from its YAML configuration."""

    raw = read_config_scalar(path, "port") or read_config_scalar(path, "http_port")
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: port must be an integer, got {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{path}: port must be between 1 and 65535, got {port}")
    return port
