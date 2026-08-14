# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for the native OCR tool."""

from pathlib import Path
from typing import Any

from xr_ai_models import OCRDetection, OCRPoint, OCRResponse
from xr_ai_tools.ocr import OCRRequest, OCRTool


class _OCR:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    async def recognize(self, image, *, merge_level="paragraph", **_kwargs):
        self.calls.append((image, merge_level))
        return OCRResponse(
            text="12.7 V",
            detections=(
                OCRDetection(
                    text="12.7 V",
                    confidence=0.98,
                    bounding_box=(OCRPoint(0.1, 0.2), OCRPoint(0.4, 0.3)),
                ),
            ),
            model="nvidia/nemotron-ocr-v2",
            raw={},
        )


async def test_ocr_tool_reads_equipment_value_from_path(tmp_path: Path) -> None:
    image = tmp_path / "meter.png"
    image.write_bytes(b"image")
    service = _OCR()
    tool = OCRTool(ocr=service)

    result = await tool.execute(OCRRequest(image=str(image), merge_level="word"))

    assert result.model_dump() == {
        "text": "12.7 V",
        "detections": [
            {
                "text": "12.7 V",
                "confidence": 0.98,
                "bounding_box": [{"x": 0.1, "y": 0.2}, {"x": 0.4, "y": 0.3}],
            }
        ],
        "model": "nvidia/nemotron-ocr-v2",
    }
    assert service.calls == [(image, "word")]


async def test_ocr_tool_model_visible_result_is_plain_text() -> None:
    tool = OCRTool(ocr=_OCR())

    result = await tool.invoke(
        '{"image":"data:image/png;base64,aW1hZ2U=","merge_level":"paragraph"}'
    )

    assert result.content == "12.7 V"
    assert result.return_direct is False


def test_ocr_tool_schema_is_strict_and_model_switching_is_injected() -> None:
    first = _OCR()
    second = _OCR()

    assert OCRTool(ocr=first).ocr is first
    assert OCRTool(ocr=second).ocr is second
    schema = OCRRequest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["merge_level"]["enum"] == [
        "word",
        "sentence",
        "paragraph",
    ]
