# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed input accepted by the web-events viewer."""

import json

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from xr_ai_runtime import Topic

_MAX_PAYLOAD_BYTES = 16 * 1024


class WebEvent(BaseModel):
    """One application event selected for the live browser view."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    topic: str = Field(min_length=1, max_length=160)
    """Stable presentation topic used to group the event in the browser."""

    payload: dict[str, JsonValue] = Field(default_factory=dict)
    """JSON-compatible application payload, limited to 16 KiB when serialized."""

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

    @field_validator("payload")
    @classmethod
    def _bound_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > _MAX_PAYLOAD_BYTES:
            raise ValueError(f"web event payload must not exceed {_MAX_PAYLOAD_BYTES} UTF-8 bytes")
        return value


WEB_EVENT_TOPIC = Topic("web-events.event", WebEvent, telemetry="none")
"""Shared runtime topic consumed by :class:`WebEventsAgent`."""


__all__ = ["WEB_EVENT_TOPIC", "WebEvent"]
