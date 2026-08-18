# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for NVIDIA Speech NIM's offline and streaming TTS APIs."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from ._protocols import TTSAudioChunk

_CHUNK_MS = 20
_SAMPLE_WIDTH = 2


class NvidiaTTSNIM:
    """Magpie NIM client that emits raw PCM while synthesis is in progress."""

    def __init__(
        self,
        base_url: str,
        *,
        language_code: str,
        voice: str,
        sample_rate: int = 22050,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        health_check: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        base = base_url.rstrip("/")
        self._offline_url = base + "/v1/audio/synthesize"
        self._stream_url = base + "/v1/audio/synthesize_online"
        self.health_url = base + "/v1/health/ready"
        self._language_code = language_code
        self._voice = voice
        self._sample_rate = sample_rate
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        self._health_check = health_check
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _form(self, text: str) -> dict[str, tuple[None, str]]:
        return {
            "text": (None, text),
            "language": (None, self._language_code),
            "voice": (None, self._voice),
            "sample_rate_hz": (None, str(self._sample_rate)),
            "encoding": (None, "LINEAR_PCM"),
        }

    async def synthesize(
        self,
        text: str,
        *,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes:
        if response_format != "wav":
            raise ValueError("NVIDIA TTS NIM offline synthesis only returns wav")
        kwargs: dict[str, Any] = {
            "files": self._form(text),
            "headers": self._headers(),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await self._client.post(self._offline_url, **kwargs)
        if response.is_error:
            logger.error(
                "NVIDIA TTS NIM {}: {}", response.status_code, response.text[:300]
            )
        response.raise_for_status()
        return response.content

    async def stream_synthesize(
        self,
        text: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[TTSAudioChunk]:
        kwargs: dict[str, Any] = {
            "files": self._form(text),
            "headers": self._headers(),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout

        frame_samples = max(1, self._sample_rate * _CHUNK_MS // 1000)
        frame_bytes = frame_samples * _SAMPLE_WIDTH
        pending = bytearray()
        async with self._client.stream("POST", self._stream_url, **kwargs) as response:
            if response.is_error:
                detail = (await response.aread()).decode(errors="replace")[:300]
                logger.error("NVIDIA TTS NIM {}: {}", response.status_code, detail)
            response.raise_for_status()
            async for block in response.aiter_bytes():
                pending.extend(block)
                while len(pending) >= frame_bytes:
                    data = bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
                    yield TTSAudioChunk(data=data, sample_rate=self._sample_rate)

        complete_bytes = len(pending) - len(pending) % _SAMPLE_WIDTH
        if complete_bytes:
            yield TTSAudioChunk(
                data=bytes(pending[:complete_bytes]),
                sample_rate=self._sample_rate,
            )

    async def health(self) -> bool:
        if not self._health_check:
            return True
        try:
            response = await self._client.get(
                self.health_url,
                headers=self._headers(),
                timeout=3.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "NvidiaTTSNIM":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()
