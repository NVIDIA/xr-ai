# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the native current- and recorded-frame vision functions."""

import base64
import io
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import (
    Builder,
    FunctionGroup,
    FunctionGroupBaseConfig,
    FunctionGroupRef,
    register_function_group,
)
from PIL import Image
from pydantic import ConfigDict, Field, ValidationError
from xr_ai_agent import FrameData, FrameSignal, FrameUnavailable, PixelFormat
from xr_ai_nat.functions.video_memory import HistoricalFrameRequest, HistoricalFrameResult
from xr_ai_nat.functions.vision import (
    HistoricalVisionRequest,
    LiveVisionRequest,
    StreamingVisionConfig,
    VisionRequest,
    VisionToolsConfig,
)
from xr_ai_nat.functions.vision._pixels import encode_image, frame_to_pil


class _Vlm:
    def __init__(self, content: str = "a blue square") -> None:
        self.content = content
        self.calls: list[tuple[Any, str, str]] = []

    async def ask_image(self, image: Any, question: str, *, system_prompt: str = ""):
        self.calls.append((image, question, system_prompt))
        return SimpleNamespace(content=self.content)

    async def stream(self, image: Any, question: str, *, system_prompt: str = ""):
        self.calls.append((image, question, system_prompt))
        for token in ("a ", "blue ", "square"):
            yield token


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


# ── recorded-frame stub group (stands in for video-memory-service) ────────────


class _VideoMemoryStubConfig(FunctionGroupBaseConfig, name="xr_video_memory_stub"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    requests: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=_VideoMemoryStubConfig)
async def _video_memory_stub(config: _VideoMemoryStubConfig, _builder: Builder):
    group = FunctionGroup(config=config)

    async def get_frame_from_time(request: HistoricalFrameRequest) -> HistoricalFrameResult:
        config.requests.append(request)
        return HistoricalFrameResult(
            path="/tmp/frame.png",
            width=1,
            height=1,
            timestamp_us=90,
            second_ago=request.second_ago,
            actual_second_ago=10.0,
        )

    group.add_function("get_frame_from_time", get_frame_from_time, description="Return a recorded frame.")
    yield group


async def _build_vision(builder: WorkflowBuilder, endpoint, vlm, requests):
    await builder.add_function_group("video_memory", _VideoMemoryStubConfig(requests=requests))
    await builder.add_function_group(
        "vision",
        VisionToolsConfig(endpoint=endpoint, vlm=vlm, video_memory=FunctionGroupRef("video_memory")),
    )
    vision = await builder.get_function_group("vision")
    return await vision.get_all_functions()


# ── _pixels frame conversion ──────────────────────────────────────────────────


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


# ── StreamingVisionConfig (live-camera streaming) ─────────────────────────────


async def test_streaming_vision_function_uses_current_participant_frame() -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("a blue square")
    config = StreamingVisionConfig(endpoint=endpoint, vlm=vlm, system_prompt="Answer briefly.")

    async with WorkflowBuilder() as builder:
        function = await builder.add_function("perception", config)
        assert endpoint.frame_callback is not None
        await endpoint.frame_callback(_seed_signal())
        chunks = [
            chunk.text
            async for chunk in function.astream(VisionRequest(participant_id="alice", query="What is shown?"))
        ]
        answer = await function.ainvoke(VisionRequest(participant_id="alice", query="What is shown?"))

    assert chunks == ["a ", "blue ", "square"]
    assert answer.text == "a blue square"
    assert answer.status == "ok"
    assert endpoint.statuses == [
        ("processing", "alice"),
        ("idle", "alice"),
        ("processing", "alice"),
        ("idle", "alice"),
    ]
    assert vlm.calls[0][1:] == ("What is shown?", "Answer briefly.")
    assert vlm.calls[0][0].startswith("data:image/jpeg;base64,")


async def test_streaming_vision_function_reports_unavailable_frame(monkeypatch) -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("unused")

    async def unavailable_frame(*_args) -> str:
        raise FrameUnavailable("No camera frame available — please try again.")

    monkeypatch.setattr("xr_ai_nat.functions.vision.functions._current_image", unavailable_frame)

    async with WorkflowBuilder() as builder:
        function = await builder.add_function("perception", StreamingVisionConfig(endpoint=endpoint, vlm=vlm))
        chunks = [
            chunk.text
            async for chunk in function.astream(VisionRequest(participant_id="alice", query="What is shown?"))
        ]
        answer = await function.ainvoke(VisionRequest(participant_id="alice", query="What is shown?"))

    assert chunks == ["No camera frame available — please try again."]
    assert answer.text == "No camera frame available — please try again."
    assert answer.status == "unavailable"
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert vlm.calls == []


# ── VisionToolsConfig — look_at_current_frame ─────────────────────────────────


async def test_look_at_current_frame_answers_from_live_frame() -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("It's a red mug.")
    requests: list[HistoricalFrameRequest] = []

    async with WorkflowBuilder() as builder:
        functions = await _build_vision(builder, endpoint, vlm, requests)
        assert set(functions) == {"vision__look_at_current_frame", "vision__look_at_past_frame"}
        look = functions["vision__look_at_current_frame"]
        await endpoint.frame_callback(_seed_signal())
        result = await look.ainvoke(LiveVisionRequest(participant_id="alice", question="What am I holding?"))

    assert result.answer == "It's a red mug."
    image, question, _system = vlm.calls[0]
    assert question == "What am I holding?"
    assert image.startswith("data:image/jpeg;base64,")


async def test_look_at_current_frame_builds_and_runs_without_video_memory() -> None:
    # A live-only consumer must be able to construct and call look_at_current_frame
    # without configuring a recorded video-memory group (lazy recorded dependency).
    endpoint = _Endpoint()
    vlm = _Vlm("It's a red mug.")

    async with WorkflowBuilder() as builder:
        await builder.add_function_group(
            "vision",
            VisionToolsConfig(endpoint=endpoint, vlm=vlm),
        )
        vision = await builder.get_function_group("vision")
        functions = await vision.get_all_functions()
        look = functions["vision__look_at_current_frame"]
        await endpoint.frame_callback(_seed_signal())
        result = await look.ainvoke(LiveVisionRequest(participant_id="alice", question="What am I holding?"))

    assert result.answer == "It's a red mug."


async def test_look_at_current_frame_reports_empty_answer_as_unavailable() -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("   ")  # blank answer → FrameUnavailable
    requests: list[HistoricalFrameRequest] = []

    async with WorkflowBuilder() as builder:
        functions = await _build_vision(builder, endpoint, vlm, requests)
        look = functions["vision__look_at_current_frame"]
        await endpoint.frame_callback(_seed_signal())
        with pytest.raises(FrameUnavailable):
            await look.ainvoke(LiveVisionRequest(participant_id="alice", question="What am I holding?"))


# ── VisionToolsConfig — look_at_past_frame ────────────────────────────────────


async def test_look_at_past_frame_uses_recorded_frame_function() -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("It was purple.")
    requests: list[HistoricalFrameRequest] = []

    async with WorkflowBuilder() as builder:
        functions = await _build_vision(builder, endpoint, vlm, requests)
        look_past = functions["vision__look_at_past_frame"]
        result = await look_past.ainvoke(
            HistoricalVisionRequest(
                participant_id="alice",
                question="What color was it?",
                second_ago=10,
                reference_time_us=100,
            )
        )

    assert result.answer == "It was purple."
    assert requests[0].reference_time_us == 100
    assert requests[0].second_ago == 10
    # The recorded PNG path was handed to the VLM as a filesystem path.
    assert vlm.calls[0][0] == Path("/tmp/frame.png")


def test_historical_vision_request_requires_a_positive_offset() -> None:
    with pytest.raises(ValidationError):
        HistoricalVisionRequest(
            participant_id="alice",
            question="What was visible?",
            second_ago=0,
            reference_time_us=100,
        )


def test_historical_vision_request_requires_a_positive_reference_time() -> None:
    # reference_time_us forwards to HistoricalFrameRequest, which requires gt=0.
    # Omitting it (no default) or passing 0 must be rejected at the boundary
    # rather than forwarding an invalid downstream request.
    with pytest.raises(ValidationError):
        HistoricalVisionRequest(
            participant_id="alice",
            question="What was visible?",
            second_ago=10,
        )
    with pytest.raises(ValidationError):
        HistoricalVisionRequest(
            participant_id="alice",
            question="What was visible?",
            second_ago=10,
            reference_time_us=0,
        )


def test_vision_request_rejects_unknown_arguments() -> None:
    with pytest.raises(ValidationError):
        VisionRequest(participant_id="alice", query="What is shown?", unsupported=True)


def test_live_vision_request_rejects_unknown_arguments() -> None:
    with pytest.raises(ValidationError):
        LiveVisionRequest(participant_id="alice", question="What is shown?", unsupported=True)
