# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tail configured JSON Lines directories into one ordered event stream."""

import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Source


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    id: int
    source_id: str
    source_title: str
    file: str
    record: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "file": self.file,
            "record": self.record,
        }


class EventStore:
    def __init__(self, max_events: int = 5000) -> None:
        self._events: deque[ActivityEvent] = deque(maxlen=max_events)
        self._next_id = 1
        self._lock = threading.Lock()

    def append(self, source: Source, path: Path, record: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(
                ActivityEvent(
                    id=self._next_id,
                    source_id=source.id,
                    source_title=source.title,
                    file=path.name,
                    record=record,
                )
            )
            self._next_id += 1

    def after(self, event_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [event.as_dict() for event in self._events if event.id > event_id]


class JsonlWatcher:
    def __init__(self, sources: tuple[Source, ...], store: EventStore) -> None:
        self.sources = sources
        self.store = store
        self._offsets: dict[Path, int] = {}

    def baseline(self) -> None:
        for source in self.sources:
            source.directory.mkdir(parents=True, exist_ok=True)
            for path in source.directory.glob("*.jsonl"):
                self._offsets[path] = path.stat().st_size

    def scan(self) -> int:
        pending: list[tuple[str, Source, Path, dict[str, Any]]] = []
        for source in self.sources:
            for path in sorted(source.directory.glob("*.jsonl")):
                pending.extend(self._read(source, path))
        pending.sort(key=lambda item: item[0])
        for _, source, path, record in pending:
            self.store.append(source, path, record)
        return len(pending)

    def _read(self, source: Source, path: Path) -> list[tuple[str, Source, Path, dict[str, Any]]]:
        offset = self._offsets.get(path, 0)
        if path.stat().st_size < offset:
            offset = 0
        records: list[tuple[str, Source, Path, dict[str, Any]]] = []
        with path.open("rb") as stream:
            stream.seek(offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                offset = stream.tell()
                record = json.loads(line)
                timestamp = str(record.get("timestamp", ""))
                records.append((timestamp, source, path, record))
        self._offsets[path] = offset
        return records


__all__ = ["ActivityEvent", "EventStore", "JsonlWatcher"]
