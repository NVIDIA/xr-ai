# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron OCR v2 HTTP adapter for both supported serving backends."""
from __future__ import annotations

import os
from typing import Literal

import httpx

from ._config import OCRSpec
from ._openai_compat import _auth_headers, _http_health, _normalize_image, _warn_if_cleartext_key
from ._protocols import ImageInput, ModelServiceError, OCRRegion


class OCRRequestError(ModelServiceError):
    """OCR input could not be read or the HTTP request failed.

    ``status_code`` is the HTTP failure status, or ``None`` for file/transport
    failures. The original exception is preserved as ``__cause__``. Requests
    are not retried automatically.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OCRResponseError(ModelServiceError, ValueError):
    """The OCR endpoint returned malformed JSON or invalid detection data.

    Also inherits ``ValueError`` to preserve existing response-validation
    handlers. Caller argument errors remain ordinary ``ValueError``/``TypeError``.
    """


def _object(value: object) -> dict:
    if not isinstance(value, dict):
        raise OCRResponseError("OCR response contains a non-object record")
    return value


def _normalized_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise OCRResponseError("OCR response confidence and coordinates must be finite numbers in [0, 1]")
    return float(value)


def _parse_regions(payload: object) -> list[OCRRegion]:
    data = _object(payload).get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise OCRResponseError("OCR response must contain exactly the requested image")
    result = _object(data[0])
    index = result.get("index")
    if type(index) is not int or index != 0:
        raise OCRResponseError("OCR response must identify the requested image at index 0")
    detections = result.get("text_detections")
    if not isinstance(detections, list):
        raise OCRResponseError("OCR response text_detections must be a list")
    regions = []
    for item in detections:
        detection = _object(item)
        prediction = _object(detection.get("text_prediction"))
        text = prediction.get("text")
        if not isinstance(text, str):
            raise OCRResponseError("OCR response text must be a string")
        confidence = _normalized_number(prediction.get("confidence"))
        points = _object(detection.get("bounding_box")).get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise OCRResponseError("OCR response polygon must contain at least three points")
        polygon = tuple(
            (_normalized_number(_object(p).get("x")), _normalized_number(_object(p).get("y")))
            for p in points
        )
        regions.append(OCRRegion(text=text, confidence=confidence, polygon=polygon))
    return regions


class NemotronOCR:
    """Client for /v1/ocr; injected HTTP clients remain caller-owned."""

    def __init__(self, spec: OCRSpec, *, client: httpx.AsyncClient | None = None):
        self._spec = spec
        self._base = spec.base_url.rstrip("/")
        self._key = os.environ.get(spec.api_key_env) if spec.api_key_env else None
        _warn_if_cleartext_key(self._base, self._key)
        self._client = client or httpx.AsyncClient(timeout=spec.timeout, trust_env=False)
        self._owns_client = client is None

    async def read_text(
        self, image: ImageInput, *, merge_level: Literal["word", "sentence", "paragraph"] = "word",
        timeout: float | None = None,
    ) -> list[OCRRegion]:
        """Return recognized text, confidence, and normalized polygons.

        Raises:
            OCRRequestError: Input-file, transport, timeout, or HTTP failure.

            OCRResponseError: The endpoint returned malformed detection data.
        """
        if merge_level not in {"word", "sentence", "paragraph"}:
            raise ValueError("merge_level must be word, sentence, or paragraph")
        try:
            url = _normalize_image(image)
        except OSError as exc:
            raise OCRRequestError("Cannot read OCR input image; check the file and permissions") from exc
        if not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
            raise ValueError("OCR requires inline PNG/JPEG data; pass bytes or a Path")
        kwargs = {} if timeout is None else {"timeout": timeout}
        try:
            response = await self._client.post(
                self._base + "/v1/ocr",
                json={"input": [{"type": "image_url", "url": url}], "merge_levels": [merge_level]},
                headers=_auth_headers(self._key), **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OCRRequestError(
                f"OCR endpoint returned HTTP {exc.response.status_code}; check the service",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise OCRRequestError("OCR request failed; check endpoint connectivity and timeout") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise OCRResponseError("OCR endpoint returned invalid JSON") from exc
        return _parse_regions(payload)

    async def health(self) -> bool:
        """Probe configured readiness; disabled probes return True."""
        return await _http_health(
            self._client, self._base + self._spec.health_path, self._spec.health_check, self._key,
        )

    async def close(self) -> None:
        """Close only the internally owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> NemotronOCR:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
