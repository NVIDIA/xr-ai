# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight image references and bounded in-process image storage."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

_IMAGE_SCHEME = "xr-image://"
ImageInput = bytes | Path | str


class ImageReference(BaseModel):
    """A lightweight reference to image bytes, a local path, or an HTTP URL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(
        min_length=1,
        description=("Opaque xr-image URI returned by an image tool, local image path, file URI, or HTTP(S) URL."),
    )

    @field_validator("uri")
    @classmethod
    def reject_inline_data(cls, value: str) -> str:
        if value.startswith("data:"):
            raise ValueError("register inline image data with ImageRegistry.put()")
        return value


class ImageRegistry:
    """Bounded in-process storage behind opaque image references."""

    def __init__(self, *, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._images: OrderedDict[str, tuple[ImageInput, str | None]] = OrderedDict()

    def put(self, image: ImageInput, *, owner: str | None = None) -> ImageReference:
        """Store image input without placing its bytes in tool results or telemetry."""

        uri = f"{_IMAGE_SCHEME}{uuid4().hex}"
        self._images[uri] = (image, owner)
        while len(self._images) > self.capacity:
            self._images.popitem(last=False)
        return ImageReference(uri=uri)

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

        parsed = urlsplit(uri)
        if parsed.scheme in {"http", "https"}:
            return uri
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme:
            raise ValueError(f"unsupported image URI scheme: {parsed.scheme}")
        return Path(uri)

    def release_owner(self, owner: str) -> None:
        """Remove opaque images associated with one participant or workflow."""

        for uri in tuple(self._images):
            if self._images[uri][1] == owner:
                del self._images[uri]

    def clear(self) -> None:
        self._images.clear()

    def __len__(self) -> int:
        return len(self._images)


__all__ = ["ImageReference", "ImageRegistry"]
