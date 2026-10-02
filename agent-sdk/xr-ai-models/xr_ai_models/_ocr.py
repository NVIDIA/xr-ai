# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron OCR v2 HTTP adapter for both supported serving backends."""
from __future__ import annotations

import math
import os
from typing import Literal

import httpx

from ._config import OCRSpec
from ._openai_compat import _auth_headers, _http_health, _normalize_image, _warn_if_cleartext_key
from ._protocols import ImageInput, OCRSpan


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
    ) -> list[OCRSpan]:
        """Return recognized text, confidence, and normalized polygons."""
        if merge_level not in {"word", "sentence", "paragraph"}:
            raise ValueError("merge_level must be word, sentence, or paragraph")
        url = _normalize_image(image)
        if not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
            raise ValueError("OCR requires inline PNG/JPEG data; pass bytes or a Path")
        kwargs = {} if timeout is None else {"timeout": timeout}
        response = await self._client.post(
            self._base + "/v1/ocr",
            json={"input": [{"type": "image_url", "url": url}], "merge_levels": [merge_level]},
            headers=_auth_headers(self._key), **kwargs,
        )
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != 1 or data[0]["index"] != 0:
            raise ValueError("OCR response must contain exactly the requested image")
        spans = []
        for item in data[0]["text_detections"]:
            prediction = item["text_prediction"]
            confidence = float(prediction["confidence"])
            polygon = tuple((float(p["x"]), float(p["y"])) for p in item["bounding_box"]["points"])
            if (not math.isfinite(confidence) or not 0 <= confidence <= 1 or len(polygon) < 3
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for point in polygon for v in point)):
                raise ValueError("OCR response contains invalid confidence or normalized polygon")
            spans.append(OCRSpan(text=prediction["text"], confidence=confidence, polygon=polygon))
        return spans

    async def health(self) -> bool:
        """Probe configured readiness; disabled probes return True."""
        return await _http_health(
            self._client, self._base + self._spec.health_path, self._spec.health_check,
        )

    async def close(self) -> None:
        """Close only the internally owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> NemotronOCR:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
