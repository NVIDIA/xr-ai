# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight image references and bounded in-process image storage."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IMAGE_SCHEME = "xr-image://"
ImageInput = bytes | Path | str


class ImageReference(BaseModel):
    """A lightweight reference to image bytes, a local path, or an HTTP URL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(
        min_length=1,
        description=("Opaque xr-image URI returned by an image tool, local image path, file URI, or HTTP(S) URL."),
    )
    """Opaque image URI, local path, file URI, or HTTP(S) URL."""

    @field_validator("uri")
    @classmethod
    def reject_inline_data(cls, value: str) -> str:
        """Reject data URIs so image bytes remain outside request payloads."""

        if value.startswith("data:"):
            raise ValueError("register inline image data with ImageRegistry.put()")
        return value


class NormalizedImagePoint(BaseModel):
    """One finite image coordinate normalized to the closed unit interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Horizontal coordinate normalized to the image width."""

    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Vertical coordinate normalized to the image height."""


class NormalizedImageBox(BaseModel):
    """An axis-aligned image box expressed in normalized coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Normalized horizontal coordinate of the left edge."""

    top: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Normalized vertical coordinate of the top edge."""

    right: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Normalized horizontal coordinate of the right edge."""

    bottom: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Normalized vertical coordinate of the bottom edge."""

    @model_validator(mode="after")
    def validate_area(self) -> NormalizedImageBox:
        """Require a non-empty rectangle."""

        if self.left >= self.right:
            raise ValueError("box left must be less than right")
        if self.top >= self.bottom:
            raise ValueError("box top must be less than bottom")
        return self


class TimedImage(BaseModel):
    """An image reference positioned on a microsecond timeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: ImageReference
    """Reference to the image content."""

    timestamp_us: int = Field(ge=0)
    """Position on the image sequence's microsecond timeline."""


class ImageRegistry:
    """Bounded in-process storage behind opaque image references."""

    def __init__(self, *, capacity: int = 128, allow_external: bool = False) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.allow_external = allow_external
        self._images: OrderedDict[str, tuple[ImageInput, str | None]] = OrderedDict()

    def put(self, image: ImageInput, *, owner: str | None = None) -> ImageReference:
        """Store image input without placing its bytes in tool results or telemetry."""

        uri = f"{_IMAGE_SCHEME}{uuid4().hex}"
        self._images[uri] = (image, owner)
        while len(self._images) > self.capacity:
            self._images.popitem(last=False)
        return ImageReference(uri=uri)

    def put_derived(
        self,
        image: ImageInput,
        *,
        source: ImageReference,
    ) -> ImageReference:
        """Store derived image input with the source reference's owner."""

        if source.uri.startswith(_IMAGE_SCHEME):
            try:
                _source_image, owner = self._images[source.uri]
            except KeyError as exc:
                raise LookupError(f"image reference is unavailable: {source.uri}") from exc
        else:
            self.resolve(source)
            owner = None
        return self.put(image, owner=owner)

    def resolve(self, reference: ImageReference) -> ImageInput:
        """Resolve an opaque handle or normalize an external image location."""

        uri = reference.uri
        if uri.startswith(_IMAGE_SCHEME):
            try:
                image, owner = self._images.pop(uri)
            except KeyError as exc:
                raise LookupError(f"image reference is unavailable: {uri}") from exc
            self._images[uri] = (image, owner)
            return image

        if not self.allow_external:
            raise ValueError("external image references are disabled")

        parsed = urlsplit(uri)
        if parsed.scheme in {"http", "https"}:
            return uri
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme:
            raise ValueError(f"unsupported image URI scheme: {parsed.scheme}")
        return Path(uri)

    def owner(self, reference: ImageReference) -> str | None:
        """Return the owner of a live opaque reference, if one is assigned."""

        if not reference.uri.startswith(_IMAGE_SCHEME):
            self.resolve(reference)
            return None
        try:
            _image, owner = self._images[reference.uri]
        except KeyError as exc:
            raise LookupError(
                f"image reference is unavailable: {reference.uri}"
            ) from exc
        return owner

    def release_owner(self, owner: str) -> None:
        """Remove opaque images associated with one participant or workflow."""

        for uri in tuple(self._images):
            if self._images[uri][1] == owner:
                del self._images[uri]

    def clear(self) -> None:
        """Remove all registered opaque images."""

        self._images.clear()

    def __len__(self) -> int:
        return len(self._images)


__all__ = [
    "ImageReference",
    "ImageRegistry",
    "NormalizedImageBox",
    "NormalizedImagePoint",
    "TimedImage",
]
