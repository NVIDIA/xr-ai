# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL storage owned by the text-memory capability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

from .schemas import TranscriptSegment, TranscriptStats

_LOGGER = logging.getLogger(__name__)


def _safe_name(source_id: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in source_id)


class TextMemoryStore:
    """Persistent timestamped text keyed by an arbitrary source identifier."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._root = self.directory.resolve()
        self._lock = Lock()

    def _check(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes text-memory directory: {path}")
        return resolved

    def _resolve_or_create(self, source_id: str) -> Path:
        safe = _safe_name(source_id)
        suffix = 1
        while True:
            stem = safe if suffix == 1 else f"{safe}_{suffix}"
            identity = self.directory / f"{stem}.identity"
            data = self.directory / f"{stem}.jsonl"
            if not identity.exists() and not data.exists():
                identity.write_text(source_id, encoding="utf-8")
                return self._check(data)
            if identity.exists() and identity.read_text(encoding="utf-8") == source_id:
                return self._check(data)
            if not identity.exists() and data.exists() and source_id == safe and suffix == 1:
                identity.write_text(source_id, encoding="utf-8")
                return self._check(data)
            suffix += 1

    def _resolve_existing(self, source_id: str) -> Path | None:
        safe = _safe_name(source_id)
        canonical_data = self.directory / f"{safe}.jsonl"
        canonical_identity = self.directory / f"{safe}.identity"
        if canonical_data.exists():
            if canonical_identity.exists():
                if canonical_identity.read_text(encoding="utf-8").strip() == source_id.strip():
                    return self._check(canonical_data)
            elif source_id == safe:
                return self._check(canonical_data)
        for identity in sorted(self.directory.glob("*.identity")):
            if identity.read_text(encoding="utf-8").strip() == source_id.strip():
                data = self._check(identity.with_suffix(".jsonl"))
                return data if data.exists() else None
        return None

    def append(self, source_id: str, timestamp_us: int, text: str) -> None:
        record = json.dumps({"timestamp_us": timestamp_us, "text": text})
        with self._lock, self._resolve_or_create(source_id).open("a", encoding="utf-8") as output:
            output.write(record + "\n")

    def query(self, source_id: str, start_us: int, end_us: int) -> list[TranscriptSegment]:
        with self._lock:
            path = self._resolve_existing(source_id)
            if path is None:
                return []
            lines = path.read_text(encoding="utf-8").splitlines()

        segments: list[TranscriptSegment] = []
        skipped = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                segment = TranscriptSegment.model_validate_json(line)
            except ValueError:
                skipped += 1
                continue
            if start_us <= segment.timestamp_us <= end_us:
                segments.append(segment)
        if skipped:
            _LOGGER.warning("Skipped %d corrupt text-memory lines in %s", skipped, path)
        return sorted(segments, key=lambda segment: segment.timestamp_us)

    def list_sources(self) -> list[str]:
        with self._lock:
            sources: list[str] = []
            identified: set[str] = set()
            for identity in sorted(self.directory.glob("*.identity")):
                sources.append(identity.read_text(encoding="utf-8"))
                identified.add(identity.stem)
            for data in sorted(self.directory.glob("*.jsonl")):
                if data.stem not in identified:
                    sources.append(data.stem)
        return sources

    def stats(self, source_id: str) -> TranscriptStats | None:
        with self._lock:
            path = self._resolve_existing(source_id)
        if path is None:
            return None
        segments = self.query(source_id, 0, 9_223_372_036_854_775_807)
        if not segments:
            return TranscriptStats(
                source_id=source_id,
                count=0,
                total_chars=0,
                earliest_us=0,
                latest_us=0,
            )
        return TranscriptStats(
            source_id=source_id,
            count=len(segments),
            total_chars=sum(len(segment.text) for segment in segments),
            earliest_us=segments[0].timestamp_us,
            latest_us=segments[-1].timestamp_us,
        )
