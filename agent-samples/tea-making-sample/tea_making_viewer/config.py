# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the sample-local activity viewer."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    title: str
    directory: Path


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    host: str
    port: int
    poll_interval_s: float
    sources: tuple[Source, ...]


def load_config(path: Path) -> ViewerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    sources = tuple(
        Source(
            id=str(item["id"]),
            title=str(item["title"]),
            directory=_resolve(base, str(item["directory"])),
        )
        for item in raw["sources"]
    )
    port = int(raw["port"])
    poll_interval_s = float(raw.get("poll_interval_s", 0.25))
    if not 1 <= port <= 65535:
        raise ValueError("viewer port must be between 1 and 65535")
    if poll_interval_s <= 0:
        raise ValueError("viewer poll_interval_s must be positive")
    if not sources or len({source.id for source in sources}) != len(sources):
        raise ValueError("viewer sources must have unique ids")
    return ViewerConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=port,
        poll_interval_s=poll_interval_s,
        sources=sources,
    )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


__all__ = ["Source", "ViewerConfig", "load_config"]
