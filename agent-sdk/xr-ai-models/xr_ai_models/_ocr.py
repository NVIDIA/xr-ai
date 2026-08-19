# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCR service adapters for NVIDIA OCR NIM and vision-language models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx
from loguru import logger

from ._openai_compat import (
    _http_health,
    _normalize_image,
    _request_headers,
    _sniff_mime,
    _to_data_url,
    _warn_if_cleartext_key,
)
from ._protocols import (
    ImageInput,
    OCRCapabilities,
    OCRDetection,
    OCRMergeLevel,
    OCRPoint,
    OCRResponse,
    VLMService,
)

_SUPPORTED_MIME_TYPES = frozenset({"image/jpeg", "image/png"})
_DEFAULT_VLM_PROMPT = (
    "Transcribe all visible text and numbers exactly in reading order. "
    "Return only the transcription."
)


class NvidiaOCR:
    """Client for the NVIDIA Image OCR NIM HTTP contract."""

    capabilities = OCRCapabilities(
        merge_levels=("word", "sentence", "paragraph"),
        structured_detections=True,
        bounding_boxes=True,
        confidence_scores=True,
        reading_order=True,
    )

    def __init__(
        self,
        base_url: str,
        model_name: str = "nvidia/nemotron-ocr-v2",
        *,
        request_path: str | None = "/v1/ocr",
        health_path: str = "/v1/health/ready",
        api_key_env: str | None = None,
        timeout: float = 60.0,
        health_check: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.request_url = base + _normalize_path(request_path)
        self.health_url = base + _normalize_path(health_path)
        self._model = model_name
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        _warn_if_cleartext_key(base_url, self._api_key)
        self._health_check = health_check
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = client is None

    async def recognize(
        self,
        image: ImageInput,
        *,
        merge_level: OCRMergeLevel = "paragraph",
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> OCRResponse:
        """Recognize text in one PNG or JPEG image using Image OCR NIM."""

        image_url = await self._data_url(image, timeout=timeout)
        payload = {
            "input": [{"type": "image_url", "url": image_url}],
            "merge_levels": [merge_level],
        }
        kwargs: dict[str, Any] = {
            "json": payload,
            "headers": _request_headers(self._api_key, headers),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await self._client.post(self.request_url, **kwargs)
        if response.is_error:
            logger.error(
                "ocr {} {}: {}",
                self._model,
                response.status_code,
                response.text[:300],
            )
        response.raise_for_status()
        return _parse_nvidia_response(response.json(), self._model)

    async def _data_url(
        self,
        image: ImageInput,
        *,
        timeout: float | None,
    ) -> str:
        if not isinstance(image, str):
            return _validated_data_url(_normalize_image(image))
        if image.startswith("data:"):
            return _validated_data_url(image)

        parsed = urlparse(image)
        if parsed.scheme in {"http", "https"}:
            kwargs: dict[str, Any] = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            response = await self._client.get(image, follow_redirects=False, **kwargs)
            response.raise_for_status()
            mime = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if mime not in _SUPPORTED_MIME_TYPES:
                mime = _sniff_mime(response.content)
            return _validated_data_url(_to_data_url(response.content, mime))

        return _validated_data_url(_normalize_image(Path(image)))

    async def health(self) -> bool:
        """Return whether the configured Image OCR endpoint is ready."""

        return await _http_health(
            self._client,
            self.health_url,
            self._health_check,
        )

    async def close(self) -> None:
        """Close the internally owned HTTP client, if any."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> NvidiaOCR:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class VLMOCR:
    """Expose any configured VLM through the OCR service protocol."""

    capabilities = OCRCapabilities(
        merge_levels=("paragraph",),
        structured_detections=False,
        bounding_boxes=False,
        confidence_scores=False,
        reading_order=False,
    )

    def __init__(self, vlm: VLMService, *, prompt: str = _DEFAULT_VLM_PROMPT) -> None:
        if not prompt.strip():
            raise ValueError("VLM OCR prompt must not be empty")
        self._vlm = vlm
        self._prompt = prompt

    async def recognize(
        self,
        image: ImageInput,
        *,
        merge_level: OCRMergeLevel = "paragraph",
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> OCRResponse:
        """Recognize text in one image through the wrapped VLM service."""

        if merge_level not in self.capabilities.merge_levels:
            raise ValueError(
                f"VLM OCR supports only {self.capabilities.merge_levels!r}, "
                f"not {merge_level!r}"
            )
        response = await self._vlm.ask_image(
            image,
            self._prompt,
            temperature=0.0,
            timeout=timeout,
            headers=headers,
        )
        text = response.content.strip()
        model = response.raw.get("model")
        return OCRResponse(
            text=text,
            detections=(),
            model=model if isinstance(model, str) else None,
            raw=response.raw,
        )

    async def health(self) -> bool:
        """Return whether the wrapped VLM endpoint is ready."""

        return await self._vlm.health()

    async def close(self) -> None:
        """Close the wrapped VLM service."""

        await self._vlm.close()

    async def __aenter__(self) -> VLMOCR:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def _normalize_path(path: str | None) -> str:
    if path is None:
        return ""
    if path and not path.startswith("/"):
        raise ValueError("OCR endpoint paths must start with '/'")
    return path.rstrip("/")


def _validated_data_url(url: str) -> str:
    prefix = url.partition(",")[0]
    if not prefix.startswith("data:") or ";base64" not in prefix:
        raise ValueError("OCR images must resolve to a base64 data URL")
    mime = prefix.removeprefix("data:").split(";", 1)[0]
    if mime not in _SUPPORTED_MIME_TYPES:
        raise ValueError("OCR images must be PNG or JPEG")
    return url


def _parse_nvidia_response(data: object, fallback_model: str) -> OCRResponse:
    if not isinstance(data, dict):
        raise ValueError("OCR response must be an object")
    raw_pages = data.get("data")
    if not isinstance(raw_pages, list) or len(raw_pages) != 1:
        raise ValueError("OCR response must contain exactly one image result")
    page = raw_pages[0]
    if not isinstance(page, dict):
        raise ValueError("OCR image result must be an object")
    raw_detections = page.get("text_detections")
    if not isinstance(raw_detections, list):
        raise ValueError("OCR image result must contain text_detections")

    detections = tuple(_parse_detection(item) for item in raw_detections)
    text = "\n".join(
        detection.text for detection in detections if detection.text
    )
    model = data.get("model", fallback_model)
    return OCRResponse(
        text=text,
        detections=detections,
        model=model if isinstance(model, str) else fallback_model,
        raw=data,
    )


def _parse_detection(data: object) -> OCRDetection:
    if not isinstance(data, dict):
        raise ValueError("OCR text detection must be an object")
    prediction = data.get("text_prediction")
    box = data.get("bounding_box")
    if not isinstance(prediction, dict) or not isinstance(box, dict):
        raise ValueError("OCR detection needs text_prediction and bounding_box")
    text = prediction.get("text")
    confidence = prediction.get("confidence")
    points = box.get("points")
    if not isinstance(text, str):
        raise ValueError("OCR text prediction must contain text")
    if not isinstance(confidence, (int, float)):
        raise ValueError("OCR text prediction must contain numeric confidence")
    if not isinstance(points, list):
        raise ValueError("OCR bounding box must contain points")
    return OCRDetection(
        text=text,
        confidence=float(confidence),
        bounding_box=tuple(_parse_point(point) for point in points),
    )


def _parse_point(data: object) -> OCRPoint:
    if not isinstance(data, dict):
        raise ValueError("OCR bounding-box point must be an object")
    x = data.get("x")
    y = data.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("OCR bounding-box point needs numeric x and y")
    return OCRPoint(x=float(x), y=float(y))


__all__ = ["NvidiaOCR", "VLMOCR"]
