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
    location: Path
    pattern: str | None
    format: str
    include_prefixes: tuple[str, ...] = ()
    exclude_events: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    host: str
    port: int
    poll_interval_s: float
    sources: tuple[Source, ...]


def load_config(path: Path, *, run_log_dir: Path | None = None) -> ViewerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    sources = tuple(_source(item, base, run_log_dir) for item in raw["sources"])
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


def _source(raw: dict[str, object], base: Path, run_log_dir: Path | None) -> Source:
    format_name = str(raw.get("format", "jsonl"))
    if format_name not in {"jsonl", "event_log"}:
        raise ValueError(f"unsupported viewer source format: {format_name}")
    directory = raw.get("directory")
    path = raw.get("path")
    if (directory is None) == (path is None):
        raise ValueError("viewer source requires exactly one of directory or path")
    if directory is not None:
        location = _resolve(base, str(directory))
        pattern = "*.jsonl"
    else:
        location = _resolve(run_log_dir or base, str(path))
        pattern = None
    return Source(
        id=str(raw["id"]),
        title=str(raw["title"]),
        location=location,
        pattern=pattern,
        format=format_name,
        include_prefixes=tuple(map(str, raw.get("include_prefixes", ()))),
        exclude_events=frozenset(map(str, raw.get("exclude_events", ()))),
    )


__all__ = ["Source", "ViewerConfig", "load_config"]
