# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed values returned by text-memory functions."""

from pydantic import BaseModel


class OperationResult(BaseModel):
    """Confirmation that a text-memory write completed."""

    ok: bool = True


class TextMemoryError(BaseModel):
    """A query error represented as data for agent and MCP callers."""

    error: str


class TranscriptSegment(BaseModel):
    """One timestamped text segment."""

    timestamp_us: int
    text: str


class TranscriptStats(BaseModel):
    """Summary statistics for one text source."""

    source_id: str
    count: int
    total_chars: int
    earliest_us: int
    latest_us: int


__all__ = ["OperationResult", "TextMemoryError", "TranscriptSegment", "TranscriptStats"]
