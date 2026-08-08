# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode configured activity sources into viewer records."""

import json
from typing import Any

from .config import Source


def decode_record(source: Source, line: bytes) -> dict[str, Any] | None:
    if source.format == "jsonl":
        return _object(json.loads(line))
    text = line.decode("utf-8")
    _, marker, payload = text.partition(" event ")
    if not marker or not payload.startswith("{"):
        return None
    record = _object(json.loads(payload))
    event = str(record.get("event", ""))
    if source.include_prefixes and not event.startswith(source.include_prefixes):
        return None
    if event in source.exclude_events:
        return None
    record.setdefault("timestamp", text[:23].replace(" ", "T", 1))
    return record


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("activity record must be a JSON object")
    return value


__all__ = ["decode_record"]
