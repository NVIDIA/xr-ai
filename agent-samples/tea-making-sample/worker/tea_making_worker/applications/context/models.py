# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for bounded participant-local application context."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextPublishRequest(_Request):
    producer: str = Field(min_length=1, max_length=80)
    topic: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=500)
    source_ref: str | None = Field(default=None, max_length=500)


class ContextQueryRequest(_Request):
    topics: tuple[str, ...] = Field(
        default=(),
        description=(
            "Only needed topics: change_watch.change, transcript.summary, or video_log.delta. "
            "Empty means all topics."
        ),
    )
    max_items: int = Field(default=3, ge=1, le=10, description="Maximum recent facts needed.")
    max_age_s: float = Field(
        default=120,
        gt=0,
        le=86_400,
        description="Oldest useful fact age in seconds.",
    )


class ContextItem(BaseModel):
    sequence: int
    producer: str
    topic: str
    summary: str
    observed_at_us: int = Field(exclude=True)
    source_ref: str | None = Field(default=None, exclude=True)


class ContextQueryResult(BaseModel):
    items: tuple[ContextItem, ...] = ()


__all__ = [
    "ContextItem",
    "ContextPublishRequest",
    "ContextQueryRequest",
    "ContextQueryResult",
]
