# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded OCR tools over opaque image and result references."""

from __future__ import annotations

import logging
from collections import OrderedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from xr_ai_models import (
    OCRCapabilities,
    OCRDetection,
    OCRMergeLevel,
    OCRResponse,
    OCRService,
)

from .image import (
    ImageReference,
    ImageRegistry,
    NormalizedImageBox,
    NormalizedImagePoint,
)
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)
_RESULT_SCHEME = "xr-ocr://"
_SUMMARY_SPAN_LIMIT = 32
_SUMMARY_TEXT_LIMIT = 4_000
_SUMMARY_SPAN_TEXT_LIMIT = 160
_DETAIL_SPAN_TEXT_LIMIT = 1_024
_MAX_PAGE_SIZE = 32


class OCRResultReference(BaseModel):
    """An opaque reference to complete OCR output held by the worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(min_length=1, description="Opaque xr-ocr URI returned by the OCR tool.")
    """Opaque URI for one live OCR result."""

    @field_validator("uri")
    @classmethod
    def require_ocr_scheme(cls, value: str) -> str:
        """Accept only handles issued by an OCR result registry."""

        if not value.startswith(_RESULT_SCHEME):
            raise ValueError("OCR result references must use the xr-ocr scheme")
        return value


class OCRResultRegistry:
    """Bounded in-process storage behind opaque OCR result references."""

    def __init__(self, *, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._results: OrderedDict[
            str, tuple[OCRResponse, str | None]
        ] = OrderedDict()

    def put(
        self,
        response: OCRResponse,
        *,
        owner: str | None = None,
    ) -> OCRResultReference:
        """Store complete structured output without provider metadata."""

        uri = f"{_RESULT_SCHEME}{uuid4().hex}"
        stored = OCRResponse(
            text=response.text,
            detections=response.detections,
            model=response.model,
            raw={},
        )
        self._results[uri] = (stored, owner)
        while len(self._results) > self.capacity:
            self._results.popitem(last=False)
        return OCRResultReference(uri=uri)

    def resolve(self, reference: OCRResultReference) -> OCRResponse:
        """Resolve a live handle and refresh its least-recently-used position."""

        try:
            response, owner = self._results.pop(reference.uri)
        except KeyError as exc:
            raise LookupError(
                f"OCR result reference is unavailable: {reference.uri}"
            ) from exc
        self._results[reference.uri] = (response, owner)
        return response

    def release_owner(self, owner: str) -> None:
        """Remove OCR results associated with one participant or workflow."""

        for uri in tuple(self._results):
            if self._results[uri][1] == owner:
                del self._results[uri]

    def clear(self) -> None:
        """Remove all registered OCR results."""

        self._results.clear()

    def __len__(self) -> int:
        return len(self._results)


class OCRRequest(StrictRequest):
    """An opaque image reference and requested OCR granularity."""

    image: ImageReference = Field(
        description="Image selected by another tool or registered by the caller."
    )
    """Image selected by another tool or registered by the caller."""

    merge_level: OCRMergeLevel = Field(
        default="paragraph",
        description="Granularity of recognized text regions.",
    )
    """Granularity used to merge neighboring recognized text regions."""


class OCRCapabilitiesResult(BaseModel):
    """Structured OCR features provided by the selected backend."""

    merge_levels: list[OCRMergeLevel]
    """Detection granularities accepted by the backend."""

    structured_spans: bool
    """Whether separately addressable recognized spans are returned."""

    polygons: bool
    """Whether recognized spans include normalized polygons and boxes."""

    confidence_scores: bool
    """Whether recognized spans include confidence scores."""

    reading_order: bool
    """Whether span identifiers follow sequential reading order."""


class OCRSpanResult(BaseModel):
    """One bounded recognized span and its normalized image geometry."""

    span_id: int = Field(ge=0)
    """Zero-based identifier in backend reading order."""

    text: str
    """Recognized text, possibly shortened to keep tool output bounded."""

    text_truncated: bool = False
    """Whether the stored span contains more text than this response."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Backend confidence score, when supplied."""

    polygon: list[NormalizedImagePoint] = Field(default_factory=list)
    """Ordered normalized vertices surrounding the recognized span."""

    box: NormalizedImageBox | None = None
    """Axis-aligned normalized bounds derived from the polygon, when available."""


class OCRResult(BaseModel):
    """Bounded OCR summary with a handle to complete worker-owned output."""

    result: OCRResultReference | None = None
    """Opaque handle for paging through the complete structured output."""

    text: str = Field(description="Bounded reading-order transcription preview.")
    """Recognized text and numbers in reading order, possibly shortened."""

    spans: list[OCRSpanResult] = Field(default_factory=list)
    """First bounded page of structured spans in reading order."""

    span_count: int = Field(default=0, ge=0)
    """Total structured spans retained behind the result handle."""

    truncated: bool = False
    """Whether more aggregate text or structured spans are available."""

    model: str | None = None
    """Backend model identifier, when supplied."""

    capabilities: OCRCapabilitiesResult
    """Structured output features provided by the selected backend."""

    available: bool = True
    """Whether the supplied image produced a usable OCR result."""

    message: str | None = None
    """Recoverable failure detail when OCR output was not produced."""


class OCRSpansRequest(StrictRequest):
    """Select one bounded page from stored structured OCR spans."""

    result: OCRResultReference
    """Opaque OCR result handle returned by ``read_image_text``."""

    offset: int = Field(default=0, ge=0)
    """Zero-based index of the first span to return."""

    limit: int = Field(default=8, ge=1, le=_MAX_PAGE_SIZE)
    """Maximum number of spans to return, capped at 32."""


class OCRSpansResult(BaseModel):
    """One bounded page of structured OCR spans."""

    result: OCRResultReference
    """Opaque OCR result handle used for this page."""

    spans: list[OCRSpanResult] = Field(default_factory=list)
    """Requested reading-order span page."""

    offset: int = Field(default=0, ge=0)
    """Zero-based index requested for this page."""

    next_offset: int | None = Field(default=None, ge=0)
    """Offset for the next page, or ``None`` after the final span."""

    span_count: int = Field(default=0, ge=0)
    """Total number of stored structured spans."""

    truncated: bool = False
    """Whether any span text was shortened in this page."""

    available: bool = True
    """Whether the referenced OCR result remains available."""

    message: str | None = None
    """Recoverable failure detail when the result handle is stale."""


class OCRTool(Tool[OCRRequest, OCRResult]):
    """Read one registered image with an injected OCR service."""

    def __init__(
        self,
        *,
        images: ImageRegistry,
        ocr: OCRService,
        results: OCRResultRegistry | None = None,
    ) -> None:
        self.images = images
        self.ocr = ocr
        self.results = results if results is not None else OCRResultRegistry()
        super().__init__(
            "read_image_text",
            (
                "Transcribe visible text and numeric equipment values from an image "
                "reference, returning bounded spans and a result handle."
            ),
            OCRRequest,
            OCRResult,
            self._read,
        )

    async def _read(self, request: OCRRequest) -> OCRResult:
        capabilities = _capabilities_result(self.ocr.capabilities)
        if request.merge_level not in self.ocr.capabilities.merge_levels:
            supported = ", ".join(self.ocr.capabilities.merge_levels)
            return OCRResult(
                text="",
                capabilities=capabilities,
                available=False,
                message=(
                    f"OCR merge level {request.merge_level!r} is unavailable; "
                    f"use one of: {supported}."
                ),
            )
        try:
            owner = self.images.owner(request.image)
            image = self.images.resolve(request.image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Image input could not be resolved: %s", error)
            return OCRResult(
                text="",
                capabilities=capabilities,
                available=False,
                message="Image input unavailable — please select it again.",
            )

        response = await self.ocr.recognize(
            image,
            merge_level=request.merge_level,
        )
        reference = self.results.put(response, owner=owner)
        text, text_truncated = _bounded_text(response.text, _SUMMARY_TEXT_LIMIT)
        spans = [
            _span_result(index, detection, _SUMMARY_SPAN_TEXT_LIMIT)
            for index, detection in enumerate(
                response.detections[:_SUMMARY_SPAN_LIMIT]
            )
        ]
        return OCRResult(
            result=reference,
            text=text,
            spans=spans,
            span_count=len(response.detections),
            truncated=(
                text_truncated
                or len(response.detections) > len(spans)
                or any(span.text_truncated for span in spans)
            ),
            model=response.model,
            capabilities=capabilities,
        )


class OCRSpansTool(Tool[OCRSpansRequest, OCRSpansResult]):
    """Page through structured spans retained by an OCR result registry."""

    def __init__(self, *, results: OCRResultRegistry) -> None:
        self.results = results
        super().__init__(
            "get_ocr_spans",
            "Retrieve one bounded page of reading-order spans from an OCR result handle.",
            OCRSpansRequest,
            OCRSpansResult,
            self._get,
        )

    def _get(self, request: OCRSpansRequest) -> OCRSpansResult:
        try:
            response = self.results.resolve(request.result)
        except LookupError as error:
            _LOGGER.warning("OCR result could not be resolved: %s", error)
            return OCRSpansResult(
                result=request.result,
                offset=request.offset,
                available=False,
                message="OCR result unavailable — please read the image again.",
            )

        detections = response.detections[request.offset : request.offset + request.limit]
        spans = [
            _span_result(request.offset + index, detection, _DETAIL_SPAN_TEXT_LIMIT)
            for index, detection in enumerate(detections)
        ]
        consumed = request.offset + len(spans)
        next_offset = consumed if consumed < len(response.detections) else None
        return OCRSpansResult(
            result=request.result,
            spans=spans,
            offset=request.offset,
            next_offset=next_offset,
            span_count=len(response.detections),
            truncated=any(span.text_truncated for span in spans),
        )


class OCRTools:
    """Own bounded OCR result state and the tools that operate on it."""

    def __init__(
        self,
        *,
        images: ImageRegistry,
        ocr: OCRService,
        result_capacity: int = 128,
    ) -> None:
        self.results = OCRResultRegistry(capacity=result_capacity)
        self.read_image_text = OCRTool(images=images, ocr=ocr, results=self.results)
        self.get_ocr_spans = OCRSpansTool(results=self.results)
        self.tools = (self.read_image_text, self.get_ocr_spans)

    def release_owner(self, owner: str) -> None:
        """Release OCR results associated with one participant or workflow."""

        self.results.release_owner(owner)

    def clear(self) -> None:
        """Release all OCR results owned by this tool group."""

        self.results.clear()


def _capabilities_result(capabilities: OCRCapabilities) -> OCRCapabilitiesResult:
    return OCRCapabilitiesResult(
        merge_levels=list(capabilities.merge_levels),
        structured_spans=capabilities.structured_detections,
        polygons=capabilities.bounding_boxes,
        confidence_scores=capabilities.confidence_scores,
        reading_order=capabilities.reading_order,
    )


def _span_result(
    span_id: int,
    detection: OCRDetection,
    text_limit: int,
) -> OCRSpanResult:
    text, text_truncated = _bounded_text(detection.text, text_limit)
    polygon = [
        NormalizedImagePoint(x=point.x, y=point.y)
        for point in detection.bounding_box
    ]
    box = _polygon_box(polygon)
    return OCRSpanResult(
        span_id=span_id,
        text=text,
        text_truncated=text_truncated,
        confidence=detection.confidence,
        polygon=polygon,
        box=box,
    )


def _polygon_box(
    polygon: list[NormalizedImagePoint],
) -> NormalizedImageBox | None:
    if not polygon:
        return None
    left = min(point.x for point in polygon)
    top = min(point.y for point in polygon)
    right = max(point.x for point in polygon)
    bottom = max(point.y for point in polygon)
    if left >= right or top >= bottom:
        return None
    return NormalizedImageBox(left=left, top=top, right=right, bottom=bottom)


def _bounded_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


__all__ = [
    "OCRCapabilitiesResult",
    "OCRRequest",
    "OCRResult",
    "OCRResultReference",
    "OCRResultRegistry",
    "OCRSpanResult",
    "OCRSpansRequest",
    "OCRSpansResult",
    "OCRSpansTool",
    "OCRTool",
    "OCRTools",
]
