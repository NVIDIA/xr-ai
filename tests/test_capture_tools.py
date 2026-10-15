# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from xr_ai_hub._capture import CAPTURE_START_TOPIC, CAPTURE_STOP_TOPIC
from xr_ai_tools.capture import CaptureTools
from xr_ai_tools.types import EmptyRequest


class _Endpoint:
    def __init__(self) -> None:
        self.messages = []

    async def send_return_data(self, message) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_capture_tools_bind_configuration_outside_model_arguments() -> None:
    endpoint = _Endpoint()
    capture = CaptureTools(
        endpoint=endpoint,
        target="sop-capture/assembly-line",
        metadata={"workflow": "wheel-install", "station": 4},
    )
    tools = capture.participant_tools("alice")

    assert [name for name, _tool in tools.items()] == [
        "start_recording",
        "stop_recording",
    ]
    start = tools.get("start_recording")
    stop = tools.get("stop_recording")
    assert start is not None
    assert stop is not None
    assert start.request_model.model_json_schema()["properties"] == {}
    assert stop.request_model.model_json_schema()["properties"] == {}

    assert await start.handler(EmptyRequest()) is None
    assert await stop.handler(EmptyRequest()) is None

    assert [message.topic for message in endpoint.messages] == [
        CAPTURE_START_TOPIC,
        CAPTURE_STOP_TOPIC,
    ]
    assert all(message.participant_id == "alice" for message in endpoint.messages)
    assert endpoint.messages[0].pts_us > 0
    assert json.loads(endpoint.messages[0].data) == {
        "target": "sop-capture/assembly-line",
        "metadata": {"workflow": "wheel-install", "station": 4},
    }
    assert endpoint.messages[1].data == b""


@pytest.mark.parametrize(
    "target",
    ["", "/absolute", "../escape", "safe/../escape", "double//slash", "bad target"],
)
def test_capture_tools_reject_unsafe_targets(target: str) -> None:
    with pytest.raises(ValueError, match="capture target"):
        CaptureTools(endpoint=_Endpoint(), target=target)


@pytest.mark.asyncio
async def test_capture_tools_snapshot_and_validate_fixed_metadata() -> None:
    endpoint = _Endpoint()
    metadata = {"workflow": "original"}
    capture = CaptureTools(endpoint=endpoint, target="captures", metadata=metadata)
    metadata["workflow"] = "changed"

    start = capture.participant_tools("alice").get("start_recording")

    assert start is not None
    await start.handler(EmptyRequest())
    assert json.loads(endpoint.messages[0].data)["metadata"] == {"workflow": "original"}
    with pytest.raises(ValueError, match="JSON-serializable"):
        CaptureTools(endpoint=endpoint, target="captures", metadata={"bad": object()})
    with pytest.raises(ValueError, match="participant_id"):
        capture.participant_tools("")
