# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small JSON Lines persistence helpers for background applications."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_records(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def session_path(output_dir: Path, participant_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    participant = re.sub(r"[^A-Za-z0-9_.-]+", "_", participant_id)
    return output_dir / f"{participant}-{stamp}.jsonl"


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


__all__ = ["append_records", "session_path", "timestamp"]
