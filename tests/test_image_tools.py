# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for image selection and image-input VLM tools."""

import base64
import io
import time
from pathlib import Path
from typing import Any, cast

import nemo_relay
import pytest
from PIL import Image
from pydantic import ValidationError
from xr_ai_hub import FrameData, FrameSignal, PixelFormat, ProcessorEndpoint
from xr_ai_models import ChatResponse, VLMService
from xr_ai_tools._pixels import encode_image, frame_to_pil
from xr_ai_tools.current_frame import (
    CurrentFrameRequest,
    CurrentFrameTool,
)
from xr_ai_tools.image import ImageReference, ImageRegistry, TimedImage
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryResult,
    ImageQueryTool,
    MultiImageQueryRequest,
    MultiImageQueryTool,
    StreamingImageQueryTool,
    VideoQueryRequest,
    VideoQueryTool,
)


class _Vlm:
    def __init__(self, content: str = "a blue square") -> None:
        self.content = content
        self.calls: list[tuple[Any, str, str, dict[str, str]]] = []
        self.stream_calls: list[tuple[Any, str, str, dict[str, str]]] = []

    async def ask_images(
        self,
        images: list[Any],
        question: str,
        *,
        system_prompt: str = "",
        headers: dict[str, str] | None = None,
    ) -> ChatResponse:
        self.calls.append((images, question, system_prompt, dict(headers or {})))
        return ChatResponse(self.content, None, None, "stop", {})

    async def stream_images(
        self,
        images: list[Any],
        question: str,
        *,
        system_prompt: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.stream_calls.append((images, question, system_prompt, dict(headers or {})))
        for token in ("a ", "blue ", "square"):
            yield token


class _Endpoint:
    def __init__(self) -> None:
        self.frame_callback = None

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

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_image_registry_bounds_handles_and_resolves_external_images(tmp_path) -> None:
    images = ImageRegistry(capacity=1)
    expired = images.put(b"first", owner="alice")
    current = images.put(b"second", owner="bob")

    with pytest.raises(LookupError, match="unavailable"):
        images.resolve(expired)
    assert images.resolve(current) == b"second"
    assert images.resolve(ImageReference(uri="https://example.com/image.png")) == ("https://example.com/image.png")
    path = tmp_path / "image.png"
    assert images.resolve(ImageReference(uri=str(path))) == path
    assert images.resolve(ImageReference(uri=path.as_uri())) == path

    images.release_owner("bob")
    assert len(images) == 0


def test_image_reference_rejects_inline_data_urls() -> None:
    with pytest.raises(ValidationError, match="ImageRegistry.put"):
        ImageReference(uri="data:image/png;base64,AAAA")


async def test_current_frame_tool_returns_an_opaque_image_without_a_vlm() -> None:
    endpoint = _Endpoint()
    images = ImageRegistry()
    tool = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=images,
    )
    signal = _seed_signal()
    await endpoint.frame_callback(signal)

    result = await tool.execute(CurrentFrameRequest(participant_id="alice"))

    assert result.image.uri.startswith("xr-image://")
    assert result.model_dump(exclude={"image"}) == {
        "width": 2,
        "height": 2,
        "timestamp_us": signal.pts_us,
        "sequence": 1,
        "participant_id": "alice",
        "track_id": "camera",
    }
    encoded = images.resolve(result.image)
    assert isinstance(encoded, bytes)
    assert Image.open(io.BytesIO(encoded)).size == (2, 2)


async def test_current_frame_tool_requests_pixels_through_real_hub(
    hub,
    make_connector,
    make_processor,
    settle,
) -> None:
    endpoint = make_processor()
    images = ImageRegistry()
    tool = CurrentFrameTool(endpoint=endpoint, images=images)
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=1)
    await settle()

    await connector.push_frame(
        bytes([20, 40, 60] * 4),
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

    result = await tool.execute(CurrentFrameRequest(participant_id="alice"))

    assert result.participant_id == "alice"
    assert isinstance(images.resolve(result.image), bytes)


async def test_image_query_consumes_selected_bytes_and_redacts_relay_events() -> None:
    images = ImageRegistry()
    image = images.put(b"jpeg bytes", owner="alice")
    vlm = _Vlm()
    tool = ImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )
    events = []
    subscriber = "image-query-events"
    intercept = "image-query-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-8"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        result = await tool.execute(ImageQueryRequest(image=image, query="What is shown?"))
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert result == ImageQueryResult(text="a blue square")
    sent_images, question, system_prompt, headers = vlm.calls[0]
    assert sent_images == [b"jpeg bytes"]
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-8"
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)
    assert all("jpeg bytes" not in event for event in llm_events)


@pytest.mark.parametrize(
    ("uri", "expected_type"),
    [
        ("https://example.com/photo.jpg", str),
        ("/tmp/photo.jpg", Path),
    ],
)
async def test_image_query_accepts_images_not_returned_by_frame_tools(
    uri: str,
    expected_type: type,
) -> None:
    images = ImageRegistry()
    vlm = _Vlm("external image")
    tool = ImageQueryTool(images=images, vlm=cast(VLMService, vlm))

    result = await tool.execute(
        ImageQueryRequest(
            image=ImageReference(uri=uri),
            query="What is in this image?",
        )
    )

    assert result.text == "external image"
    assert isinstance(vlm.calls[0][0][0], expected_type)


async def test_multi_image_query_preserves_caller_order() -> None:
    images = ImageRegistry()
    first = images.put(b"first")
    second = images.put(b"second")
    vlm = _Vlm("changed from empty to full")
    tool = MultiImageQueryTool(images=images, vlm=cast(VLMService, vlm))

    result = await tool.execute(
        MultiImageQueryRequest(
            images=[first, second],
            query="What changed?",
        )
    )

    assert result.text == "changed from empty to full"
    assert vlm.calls[0][:2] == ([b"first", b"second"], "What changed?")


async def test_video_query_supplies_ordered_images_and_timestamp_context() -> None:
    images = ImageRegistry()
    first = TimedImage(image=images.put(b"first"), timestamp_us=1_000_000)
    second = TimedImage(image=images.put(b"second"), timestamp_us=2_500_000)
    vlm = _Vlm("the cup was filled")
    tool = VideoQueryTool(images=images, vlm=cast(VLMService, vlm))

    result = await tool.execute(
        VideoQueryRequest(
            frames=[first, second],
            query="What happened?",
        )
    )

    assert result.text == "the cup was filled"
    sent_images, question, _system_prompt, _headers = vlm.calls[0]
    assert sent_images == [b"first", b"second"]
    assert "frame 1: timestamp_us=1000000, offset_s=0.000000" in question
    assert "frame 2: timestamp_us=2500000, offset_s=1.500000" in question
    assert question.endswith("What happened?")


def test_video_query_requires_chronological_frames() -> None:
    image = ImageReference(uri="https://example.com/frame.jpg")
    with pytest.raises(ValidationError, match="chronological"):
        VideoQueryRequest(
            frames=[
                TimedImage(image=image, timestamp_us=2),
                TimedImage(image=image, timestamp_us=1),
            ],
            query="What happened?",
        )


async def test_image_query_hides_reasoning_and_marks_failures_unavailable() -> None:
    images = ImageRegistry()
    image = images.put(b"image")
    reasoning = ImageQueryTool(
        images=images,
        vlm=cast(VLMService, _Vlm("<think>inspect</think>  a red mug  ")),
    )
    assert (await reasoning.execute(ImageQueryRequest(image=image, query="What?"))).text == "a red mug"

    class FailingVlm:
        async def ask_images(self, *_args, **_kwargs):
            raise RuntimeError("VLM failed")

    failure = ImageQueryTool(
        images=images,
        vlm=cast(VLMService, FailingVlm()),
    )
    result = await failure.execute(ImageQueryRequest(image=image, query="What?"))
    assert result == ImageQueryResult(
        text="VLM server unavailable — please retry.",
        available=False,
    )


async def test_streaming_image_query_yields_typed_chunks() -> None:
    images = ImageRegistry()
    image = images.put(b"image")
    vlm = _Vlm()
    tool = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )

    chunks = [chunk async for chunk in tool.stream(ImageQueryRequest(image=image, query="What is shown?"))]

    assert [chunk.text for chunk in chunks] == ["a ", "blue ", "square"]
    assert vlm.stream_calls[0][:3] == ([b"image"], "What is shown?", "Answer briefly.")


async def test_streaming_image_query_stops_after_partial_failure() -> None:
    class PartialFailureVlm:
        async def stream_images(self, *_args, **_kwargs):
            yield "The object is "
            raise RuntimeError("stream disconnected")

    images = ImageRegistry()
    tool = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, PartialFailureVlm()),
    )

    chunks = [chunk.text async for chunk in tool.stream(ImageQueryRequest(image=images.put(b"image"), query="What?"))]

    assert chunks == ["The object is "]


async def test_streaming_image_query_reports_failure_before_output() -> None:
    class ImmediateFailureVlm:
        async def stream_images(self, *_args, **_kwargs):
            if False:
                yield ""
            raise RuntimeError("stream unavailable")

    images = ImageRegistry()
    tool = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, ImmediateFailureVlm()),
    )

    chunks = [chunk.text async for chunk in tool.stream(ImageQueryRequest(image=images.put(b"image"), query="What?"))]

    assert chunks == ["VLM server unavailable — please retry."]
