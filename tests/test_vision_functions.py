# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for native current- and recorded-frame vision tools."""

import base64
import io
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_tools._pixels import encode_image, frame_to_pil
from xr_ai_tools.historical_vision import HistoricalVisionRequest, HistoricalVisionTool
from xr_ai_tools.live_vision import LiveVisionTool, VisionRequest
from xr_ai_tools.tools import Tool
from xr_ai_tools.video_memory import HistoricalFrameRequest, HistoricalFrameResult


class _Vlm:
    def __init__(self, content: str = "a blue square") -> None:
        self.content = content
        self.calls: list[tuple[Any, str, str]] = []

    async def ask_image(
        self,
        image: Any,
        question: str,
        *,
        system_prompt: str = "",
        headers: dict[str, str] | None = None,
    ):
        del headers
        self.calls.append((image, question, system_prompt))
        return SimpleNamespace(content=self.content)


class _Endpoint:
    def __init__(self) -> None:
        self.frame_callback = None
        self.statuses: list[tuple[str, str]] = []

    def on_frame(self, callback) -> None:
        self.frame_callback = callback

    def on_participant(self, _callback) -> None:
        pass

    async def request_frame(self, signal: FrameSignal) -> FrameData:
        return FrameData(
            seq=signal.seq,
            pts_us=signal.pts_us,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data=bytes([20, 40, 60] * 4),
            participant_id=signal.participant_id,
            track_id=signal.track_id,
        )

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))


def _seed_signal(participant_id: str = "alice") -> FrameSignal:
    return FrameSignal(
        slot=0,
        seq=1,
        pts_us=time.time_ns() // 1_000,
        width=2,
        height=2,
        fmt=PixelFormat.RGB24,
        data_sz=12,
        participant_id=participant_id,
        track_id="camera",
    )


@pytest.mark.parametrize(
    ("pixel_format", "data"),
    [
        (PixelFormat.RGBA, bytes([20, 40, 60, 255] * 4)),
        (PixelFormat.BGRA, bytes([60, 40, 20, 255] * 4)),
        (PixelFormat.I420, bytes([80] * 4 + [128, 128])),
        (PixelFormat.NV12, bytes([80] * 4 + [128, 128])),
    ],
)
def test_frame_to_pil_supports_non_rgb_frame_formats(pixel_format, data) -> None:
    frame = FrameData(
        seq=1,
        pts_us=0,
        width=2,
        height=2,
        fmt=pixel_format,
        data=data,
        participant_id="alice",
        track_id="camera",
    )

    image_url = encode_image(frame_to_pil(frame))
    image = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1])))

    assert image_url.startswith("data:image/jpeg;base64,")
    assert image.mode == "RGB"
    assert image.size == (2, 2)


async def test_live_vision_builds_without_video_memory_and_answers_current_frame() -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("It's a red mug.")
    tool = LiveVisionTool(endpoint=endpoint, vlm=vlm)
    await endpoint.frame_callback(_seed_signal())

    result = await tool.execute(VisionRequest(participant_id="alice", query="What am I holding?"))

    assert result.text == "It's a red mug."
    assert result.available is True
    image, question, _system = vlm.calls[0]
    assert question == "What am I holding?"
    assert image.startswith("data:image/jpeg;base64,")
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]


async def test_live_vision_requests_pixels_through_real_hub(hub, make_connector, make_processor, settle) -> None:
    endpoint = make_processor()
    tool = LiveVisionTool(endpoint=endpoint, vlm=_Vlm("hub frame"), manage_status=False)
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=1)
    await settle()

    pixels = bytes([20, 40, 60] * 4)
    await connector.push_frame(
        pixels,
        2,
        2,
        PixelFormat.RGB24,
        time.time_ns() // 1_000,
        "alice",
        "camera",
    )
    for _ in range(20):
        if tool.frames.participants() == ["alice"]:
            break
        await settle()

    result = await tool.execute(VisionRequest(participant_id="alice", query="What is visible?"))

    assert result.model_dump() == {"text": "hub frame", "available": True}


async def test_live_vision_marks_vlm_failure_unavailable_and_idle() -> None:
    class FailingVlm:
        async def ask_image(self, *_args, **_kwargs):
            raise RuntimeError("VLM failed")

    endpoint = _Endpoint()
    tool = LiveVisionTool(endpoint=endpoint, vlm=FailingVlm())
    await endpoint.frame_callback(_seed_signal())

    result = await tool.execute(VisionRequest(participant_id="alice", query="What am I holding?"))

    assert result.available is False
    assert result.text == "VLM server unavailable — please retry."
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]


async def test_live_vision_hides_model_reasoning() -> None:
    endpoint = _Endpoint()
    tool = LiveVisionTool(
        endpoint=endpoint,
        vlm=_Vlm("<think>inspect pixels</think>\n  a red mug  "),
    )
    await endpoint.frame_callback(_seed_signal())

    result = await tool.execute(VisionRequest(participant_id="alice", query="What am I holding?"))

    assert result.text == "a red mug"


async def test_live_vision_marks_empty_answer_unavailable() -> None:
    endpoint = _Endpoint()
    tool = LiveVisionTool(endpoint=endpoint, vlm=_Vlm("   "))
    await endpoint.frame_callback(_seed_signal())

    result = await tool.execute(VisionRequest(participant_id="alice", query="What am I holding?"))

    assert result.available is False
    assert "did not produce an answer" in result.text


async def test_historical_vision_uses_recorded_frame_tool() -> None:
    requests: list[HistoricalFrameRequest] = []

    async def get_frame(request: HistoricalFrameRequest) -> HistoricalFrameResult:
        requests.append(request)
        return HistoricalFrameResult(
            path="/tmp/frame.png",
            width=1,
            height=1,
            timestamp_us=90,
            second_ago=request.second_ago,
            actual_second_ago=10.0,
        )

    video = SimpleNamespace(
        get_frame_from_time=Tool(
            "get_frame_from_time",
            "Return a recorded frame.",
            HistoricalFrameRequest,
            HistoricalFrameResult,
            get_frame,
        )
    )
    vlm = _Vlm("<think>check the frame</think>\n It was purple. ")
    tool = HistoricalVisionTool(video=video, vlm=vlm)

    result = await tool.execute(
        HistoricalVisionRequest(
            participant_id="alice",
            query="What color was it?",
            second_ago=10,
            reference_time_us=100,
        )
    )

    assert result.text == "It was purple."
    assert requests[0].reference_time_us == 100
    assert requests[0].second_ago == 10
    assert vlm.calls[0][0] == Path("/tmp/frame.png")


@pytest.mark.parametrize(
    "arguments",
    [
        {"participant_id": "alice", "query": "What was visible?", "second_ago": 0, "reference_time_us": 100},
        {"participant_id": "alice", "query": "What was visible?", "second_ago": 10},
        {"participant_id": "alice", "query": "What was visible?", "second_ago": 10, "reference_time_us": 0},
    ],
)
def test_historical_vision_request_rejects_invalid_time(arguments) -> None:
    with pytest.raises(ValidationError):
        HistoricalVisionRequest(**arguments)


def test_live_vision_request_rejects_unknown_arguments() -> None:
    with pytest.raises(ValidationError):
        VisionRequest(
            participant_id="alice",
            query="What is shown?",
            unsupported=True,
        )
