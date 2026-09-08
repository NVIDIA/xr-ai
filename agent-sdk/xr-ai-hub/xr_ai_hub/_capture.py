# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped, request-driven client image capture."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import uuid4

from ._processor import ProcessorEndpoint
from ._types import ImageCaptureCancel, ImageCaptureData, ImageCaptureRequest


class ImageCaptureUnavailable(RuntimeError):
    """Raised when a client image capture does not complete successfully."""


class ClientImageCaptureSource:
    """Request encoded still images from participant clients through the hub."""

    def __init__(
        self,
        endpoint: ProcessorEndpoint,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("image capture timeout must be positive")
        self._endpoint = endpoint
        self._timeout_s = timeout_s
        self._pending: dict[
            tuple[str, str], asyncio.Future[ImageCaptureData]
        ] = {}
        self._unsubscribe = endpoint.on_image_capture(self._on_image)
        endpoint.on_participant(self._on_participant)

    async def capture(self, participant_id: str) -> ImageCaptureData:
        """Ask a connected participant for one still image."""

        if not participant_id.strip():
            raise ValueError("image capture requires a participant")
        if participant_id not in self._endpoint.connected_participants:
            raise ImageCaptureUnavailable("Participant is not connected.")

        request_id = uuid4().hex
        key = (participant_id, request_id)
        future: asyncio.Future[ImageCaptureData] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[key] = future
        request = ImageCaptureRequest(
            participant_id=participant_id,
            request_id=request_id,
            timeout_ms=max(1, int(self._timeout_s * 1_000)),
        )
        try:
            await self._endpoint.request_image_capture(request)
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future), timeout=self._timeout_s
                )
            except asyncio.TimeoutError as exc:
                raise ImageCaptureUnavailable(
                    "The client did not return a picture before the timeout."
                ) from exc
        finally:
            self._pending.pop(key, None)
            if not future.done():
                future.cancel()
                with suppress(Exception):
                    await self._endpoint.cancel_image_capture(
                        ImageCaptureCancel(
                            participant_id=participant_id,
                            request_id=request_id,
                        )
                    )

    async def _on_image(self, image: ImageCaptureData) -> None:
        future = self._pending.get((image.participant_id, image.request_id))
        if future is not None and not future.done():
            future.set_result(image)

    async def _on_participant(self, event) -> None:
        if event.joined:
            return
        for (participant_id, _request_id), future in tuple(self._pending.items()):
            if participant_id == event.participant_id and not future.done():
                future.set_exception(
                    ImageCaptureUnavailable("Participant disconnected during capture.")
                )

    def close(self) -> None:
        """Detach the capture callback and fail outstanding requests."""

        self._unsubscribe()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ImageCaptureUnavailable("Image capture closed."))
        self._pending.clear()


__all__ = ["ClientImageCaptureSource", "ImageCaptureUnavailable"]
