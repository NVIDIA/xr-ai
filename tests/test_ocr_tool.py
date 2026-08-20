# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for bounded, registry-backed OCR tools."""

import json
from typing import Any

import pytest
from pydantic import ValidationError
from xr_ai_models import (
    OCRCapabilities,
    OCRDetection,
    OCRPoint,
    OCRResponse,
)
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.ocr import (
    OCRRequest,
    OCRResultRegistry,
    OCRSpansRequest,
    OCRSpansTool,
    OCRTool,
    OCRTools,
)


class _OCR:
    capabilities = OCRCapabilities(
        merge_levels=("word", "sentence", "paragraph"),
        structured_detections=True,
        bounding_boxes=True,
        confidence_scores=True,
        reading_order=True,
    )

    def __init__(self, response: OCRResponse | None = None) -> None:
        self.calls: list[tuple[Any, str]] = []
        self.response = response or OCRResponse(
            text="12.7 V",
            detections=(
                OCRDetection(
                    text="12.7 V",
                    confidence=0.98,
                    bounding_box=(
                        OCRPoint(0.1, 0.2),
                        OCRPoint(0.4, 0.2),
                        OCRPoint(0.4, 0.3),
                        OCRPoint(0.1, 0.3),
                    ),
                ),
            ),
            model="nvidia/nemotron-ocr-v2",
            raw={"provider_only": "not exposed"},
        )

    async def recognize(self, image, *, merge_level="paragraph", **_kwargs):
        self.calls.append((image, merge_level))
        return self.response


async def test_ocr_tool_reads_registered_image_and_returns_bounded_geometry() -> None:
    images = ImageRegistry()
    image = images.put(b"image", owner="alice")
    service = _OCR()
    tool = OCRTool(images=images, ocr=service)

    result = await tool.execute(OCRRequest(image=image, merge_level="word"))

    assert result.result is not None
    assert result.result.uri.startswith("xr-ocr://")
    assert result.model_dump(exclude={"result"}) == {
        "text": "12.7 V",
        "spans": [
            {
                "span_id": 0,
                "text": "12.7 V",
                "text_truncated": False,
                "confidence": 0.98,
                "polygon": [
                    {"x": 0.1, "y": 0.2},
                    {"x": 0.4, "y": 0.2},
                    {"x": 0.4, "y": 0.3},
                    {"x": 0.1, "y": 0.3},
                ],
                "box": {"left": 0.1, "top": 0.2, "right": 0.4, "bottom": 0.3},
            }
        ],
        "span_count": 1,
        "truncated": False,
        "model": "nvidia/nemotron-ocr-v2",
        "capabilities": {
            "merge_levels": ["word", "sentence", "paragraph"],
            "structured_spans": True,
            "polygons": True,
            "confidence_scores": True,
            "reading_order": True,
        },
        "available": True,
        "message": None,
    }
    assert service.calls == [(b"image", "word")]
    assert tool.results.resolve(result.result).raw == {}


async def test_ocr_tool_model_visible_result_uses_handles_not_image_or_raw_data() -> None:
    images = ImageRegistry()
    image = images.put(b"secret image")
    tool = OCRTool(images=images, ocr=_OCR())

    invocation = await tool.invoke(json.dumps({"image": image.model_dump()}))
    content = json.loads(invocation.content)

    assert content["text"] == "12.7 V"
    assert content["result"]["uri"].startswith("xr-ocr://")
    assert "secret image" not in invocation.content
    assert "provider_only" not in invocation.content
    assert invocation.return_direct is False


async def test_ocr_tool_returns_unavailable_for_a_stale_image() -> None:
    images = ImageRegistry(capacity=1)
    stale = images.put(b"first")
    images.put(b"second")
    service = _OCR()
    tool = OCRTool(images=images, ocr=service)

    result = await tool.execute(OCRRequest(image=stale))

    assert result.available is False
    assert result.result is None
    assert result.message == "Image input unavailable — please select it again."
    assert service.calls == []


async def test_ocr_tool_reports_an_unsupported_backend_merge_level() -> None:
    images = ImageRegistry()
    image = images.put(b"image")
    service = _OCR()
    service.capabilities = OCRCapabilities(
        merge_levels=("paragraph",),
        structured_detections=False,
        bounding_boxes=False,
        confidence_scores=False,
        reading_order=False,
    )
    tool = OCRTool(images=images, ocr=service)

    result = await tool.execute(OCRRequest(image=image, merge_level="word"))

    assert result.available is False
    assert result.capabilities.structured_spans is False
    assert result.message == (
        "OCR merge level 'word' is unavailable; use one of: paragraph."
    )
    assert service.calls == []


def test_ocr_request_rejects_inline_data_and_extra_arguments() -> None:
    with pytest.raises(ValidationError, match="ImageRegistry.put"):
        OCRRequest(image={"uri": "data:image/png;base64,aW1hZ2U="})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OCRRequest(image={"uri": "xr-image://source"}, query="ignore")


async def test_ocr_summary_caps_spans_and_text_then_pages_stored_output() -> None:
    detections = tuple(
        OCRDetection(
            text=f"span-{index}-" + "x" * 200,
            bounding_box=(OCRPoint(0.1, 0.1), OCRPoint(0.2, 0.2)),
        )
        for index in range(40)
    )
    response = OCRResponse(
        text="z" * 5_000,
        detections=detections,
        model="large-output",
        raw={},
    )
    images = ImageRegistry()
    image = images.put(b"image")
    group = OCRTools(images=images, ocr=_OCR(response))

    summary = await group.read_image_text.execute(OCRRequest(image=image))

    assert len(summary.text) == 4_000
    assert len(summary.spans) == 32
    assert all(len(span.text) == 160 for span in summary.spans)
    assert summary.span_count == 40
    assert summary.truncated is True
    assert summary.result is not None

    page = await group.get_ocr_spans.execute(
        OCRSpansRequest(result=summary.result, offset=32, limit=8)
    )
    assert [span.span_id for span in page.spans] == list(range(32, 40))
    assert all(len(span.text) == 208 for span in page.spans)
    assert page.next_offset is None
    assert page.span_count == 40


async def test_ocr_results_are_bounded_and_released_with_the_image_owner() -> None:
    images = ImageRegistry()
    first_image = images.put(b"first", owner="alice")
    second_image = images.put(b"second", owner="bob")
    group = OCRTools(images=images, ocr=_OCR(), result_capacity=1)

    first = await group.read_image_text.execute(OCRRequest(image=first_image))
    second = await group.read_image_text.execute(OCRRequest(image=second_image))
    assert first.result is not None
    assert second.result is not None

    expired = await group.get_ocr_spans.execute(OCRSpansRequest(result=first.result))
    assert expired.available is False

    group.release_owner("bob")
    released = await group.get_ocr_spans.execute(OCRSpansRequest(result=second.result))
    assert released.available is False


def test_ocr_service_and_result_registry_are_injected() -> None:
    images = ImageRegistry()
    first = _OCR()
    second = _OCR()
    results = OCRResultRegistry()

    assert OCRTool(images=images, ocr=first, results=results).ocr is first
    assert OCRTool(images=images, ocr=second, results=results).ocr is second
    assert OCRTool(images=images, ocr=first, results=results).results is results
    assert OCRSpansTool(results=results).results is results
    schema = OCRRequest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["merge_level"]["enum"] == [
        "word",
        "sentence",
        "paragraph",
    ]
