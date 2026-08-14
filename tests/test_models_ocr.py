# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire and adapter coverage for OCR model services."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from xr_ai_models import (
    VLMOCR,
    Capabilities,
    ChatResponse,
    NvidiaOCR,
    OCRDetection,
    OCRPoint,
    OCRService,
    load_models_config_from_dict,
    make_ocr,
)

_PNG = b"\x89PNG\r\n\x1a\nocr-test"
_OCR_RESPONSE = {
    "model": "nvidia/nemotron-ocr-v2",
    "data": [
        {
            "index": 0,
            "text_detections": [
                {
                    "text_prediction": {"text": "12.7 V", "confidence": 0.98},
                    "bounding_box": {
                        "points": [
                            {"x": 0.1, "y": 0.2},
                            {"x": 0.4, "y": 0.2},
                            {"x": 0.4, "y": 0.3},
                            {"x": 0.1, "y": 0.3},
                        ]
                    },
                }
            ],
        }
    ],
    "usage": {"images_size_mb": 0.01},
}


class _OCRStub:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/meter.png":
            return httpx.Response(200, content=_PNG, headers={"content-type": "image/png"})
        if request.method == "GET":
            return httpx.Response(200, json={"ready": True})
        return httpx.Response(200, json=_OCR_RESPONSE)


async def test_nvidia_ocr_sends_nim_payload_and_parses_detections(monkeypatch) -> None:
    monkeypatch.setenv("OCR_API_KEY", "nvapi-test")
    stub = _OCRStub()
    client = stub.client()
    ocr = NvidiaOCR(
        "https://ocr.example.test",
        api_key_env="OCR_API_KEY",
        client=client,
    )
    try:
        response = await ocr.recognize(
            _PNG,
            merge_level="word",
            headers={"x-relay-session": "session-1"},
        )
    finally:
        await client.aclose()

    request = stub.requests[-1]
    assert str(request.url) == "https://ocr.example.test/v1/ocr"
    assert request.headers["Authorization"] == "Bearer nvapi-test"
    assert request.headers["x-relay-session"] == "session-1"
    payload = json.loads(request.content)
    assert payload == {
        "input": [
            {
                "type": "image_url",
                "url": "data:image/png;base64," + base64.b64encode(_PNG).decode(),
            }
        ],
        "merge_levels": ["word"],
    }
    assert response.text == "12.7 V"
    assert response.detections == (
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
    )


async def test_nvidia_ocr_downloads_http_images_without_leaking_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OCR_API_KEY", "nvapi-test")
    stub = _OCRStub()
    client = stub.client()
    ocr = NvidiaOCR(
        "https://ocr.example.test",
        api_key_env="OCR_API_KEY",
        client=client,
    )
    try:
        await ocr.recognize("https://images.example.test/meter.png")
    finally:
        await client.aclose()

    image_request, ocr_request = stub.requests
    assert "Authorization" not in image_request.headers
    assert ocr_request.headers["Authorization"] == "Bearer nvapi-test"


async def test_nvidia_ocr_accepts_hosted_full_invoke_url() -> None:
    stub = _OCRStub()
    client = stub.client()
    ocr = NvidiaOCR(
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2",
        request_path=None,
        health_check=False,
        client=client,
    )
    try:
        assert await ocr.health() is True
        await ocr.recognize(_PNG)
    finally:
        await client.aclose()

    assert len(stub.requests) == 1
    assert str(stub.requests[0].url) == (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"
    )


async def test_nvidia_ocr_reads_local_path_and_rejects_unsupported_images(
    tmp_path: Path,
) -> None:
    stub = _OCRStub()
    client = stub.client()
    ocr = NvidiaOCR("http://localhost:8000", client=client)
    png = tmp_path / "meter.png"
    png.write_bytes(_PNG)
    gif = tmp_path / "meter.gif"
    gif.write_bytes(b"GIF89a")
    try:
        result = await ocr.recognize(str(png))
        with pytest.raises(ValueError, match="PNG or JPEG"):
            await ocr.recognize(gif)
    finally:
        await client.aclose()

    assert result.text == "12.7 V"


class _VLM:
    capabilities = Capabilities(vision=True)

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str, dict[str, Any]]] = []
        self.closed = False

    async def ask_image(self, image, question, **kwargs):
        self.calls.append((image, question, kwargs))
        return ChatResponse(
            content="  220 V  ",
            reasoning=None,
            tool_calls=None,
            finish_reason="stop",
            raw={"model": "replacement-vlm"},
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


async def test_vlm_ocr_adapts_any_vlm_to_the_same_protocol() -> None:
    vlm = _VLM()
    ocr = VLMOCR(vlm, prompt="Read this meter.")

    result = await ocr.recognize(_PNG, merge_level="word", timeout=4.0)
    await ocr.close()

    assert result.text == "220 V"
    assert result.detections == (OCRDetection(text="220 V"),)
    assert result.model == "replacement-vlm"
    assert vlm.calls == [
        (_PNG, "Read this meter.", {"temperature": 0.0, "timeout": 4.0, "headers": None})
    ]
    assert vlm.closed is True
    assert isinstance(ocr, OCRService)


async def test_make_ocr_switches_backends_from_the_model_profile() -> None:
    nvidia_config = load_models_config_from_dict(
        {
            "ocr": {
                "kind": "preset:nemotron_ocr_v2",
                "base_url": "http://localhost:8000",
            }
        }
    )
    nvidia = make_ocr(nvidia_config, "ocr")

    vlm_config = load_models_config_from_dict(
        {
            "ocr": {
                "category": "ocr",
                "kind": "openai_compat",
                "base_url": "http://localhost:8100",
                "model_name": "vlm",
                "capabilities": {"vision": True},
                "prompt": "Read the display.",
            }
        }
    )
    vlm = make_ocr(vlm_config, "ocr")
    try:
        assert isinstance(nvidia, NvidiaOCR)
        assert isinstance(vlm, VLMOCR)
    finally:
        await nvidia.close()
        await vlm.close()


def test_hosted_profile_can_override_the_preset_request_path() -> None:
    config = load_models_config_from_dict(
        {
            "models": {
                "ocr": {
                    "adapter": {
                        "preset": "nemotron_ocr_v2",
                        "request_path": None,
                    },
                    "endpoint": {
                        "base_url": "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2",
                        "api_key_env": "NGC_API_KEY",
                        "readiness": "none",
                    },
                    "deployment": {"ownership": "external"},
                }
            }
        }
    )

    spec = config.ocr("ocr")
    assert spec.adapter.request_path is None
    assert spec.endpoint.health_check is False
    assert config.required_credentials == ("NGC_API_KEY",)
