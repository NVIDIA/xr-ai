# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed input accepted by the web-events viewer."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from xr_ai_runtime import Topic


class WebEvent(BaseModel):
    """One application event selected for the live browser view."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=160)
    """Stable presentation topic used to group the event in the browser."""

    payload: dict[str, JsonValue] = Field(default_factory=dict)
    """JSON-compatible application payload rendered by the browser."""

    title: str | None = Field(default=None, min_length=1, max_length=120)
    """Optional human-readable title for the presentation topic."""

    @field_validator("topic", "title")
    @classmethod
    def _strip_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("web event topic and title must not be blank")
        return stripped


WEB_EVENT_TOPIC = Topic("web-events.event", WebEvent, telemetry="none")
"""Shared runtime topic consumed by :class:`WebEventsAgent`."""


__all__ = ["WEB_EVENT_TOPIC", "WebEvent"]
