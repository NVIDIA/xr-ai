# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small per-participant JSON Lines transcript store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jsonl import append_records, session_path, timestamp


@dataclass(slots=True)
class TranscriptState:
    path: Path
    next_summary: float
    turns: list[str] = field(default_factory=list)
    writes: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True
    summarizing: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def transcript_path(output_dir: Path, participant_id: str) -> Path:
    return session_path(output_dir, participant_id)


__all__ = ["TranscriptState", "append_records", "timestamp", "transcript_path"]
