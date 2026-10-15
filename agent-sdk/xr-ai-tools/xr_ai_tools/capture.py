# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-bound tools for agent-controlled media capture."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from xr_ai_hub import DataMessage, ProcessorEndpoint
from xr_ai_hub._capture import CAPTURE_START_TOPIC, CAPTURE_STOP_TOPIC

from .tools import Tool, ToolSet
from .types import EmptyRequest

_MAX_TARGET_COMPONENT = 96


def _validate_target(target: str) -> str:
    if not isinstance(target, str) or not target or target.startswith("/"):
        raise ValueError("capture target must be a non-empty relative path")
    parts = target.split("/")
    if any(
        not part
        or part in {".", ".."}
        or len(part) > _MAX_TARGET_COMPONENT
        or any(not (char.isalnum() or char in "-_.") for char in part)
        for part in parts
    ):
        raise ValueError(
            "capture target components may contain only letters, numbers, '.', '-', and '_'"
        )
    return "/".join(parts)


class CaptureTools:
    """Create parameterless recording tools with wrapper-owned configuration.

    The application chooses the destination namespace and immutable metadata
    when it constructs this wrapper. ``participant_tools`` then binds the two
    model-visible operations to one participant, so neither operation accepts
    a filesystem path, participant id, or metadata from the model.
    """

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        target: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._endpoint = endpoint
        target = _validate_target(target)
        try:
            self._start_payload = json.dumps(
                {"target": target, "metadata": dict(metadata or {})},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("capture metadata must be JSON-serializable") from exc

    def participant_tools(self, participant_id: str) -> ToolSet:
        """Return start and stop tools bound to one participant."""

        if not isinstance(participant_id, str) or not participant_id:
            raise ValueError("participant_id must not be empty")

        async def start(_request: EmptyRequest) -> None:
            await self._send(participant_id, CAPTURE_START_TOPIC, self._start_payload)

        async def stop(_request: EmptyRequest) -> None:
            await self._send(participant_id, CAPTURE_STOP_TOPIC, b"")

        return ToolSet((
            Tool(
                "start_recording",
                "Start a new timestamped audio, video, and transcript recording.",
                EmptyRequest,
                None,
                start,
            ),
            Tool(
                "stop_recording",
                "Stop and finalize the active audio, video, and transcript recording.",
                EmptyRequest,
                None,
                stop,
            ),
        ))

    async def _send(self, participant_id: str, topic: str, data: bytes) -> None:
        await self._endpoint.send_return_data(DataMessage(
            participant_id=participant_id,
            topic=topic,
            pts_us=time.time_ns() // 1_000,
            data=data,
        ))


__all__ = ["CaptureTools"]
