# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thin async clients for stt-server and tts-server, plus generic readiness probes.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import httpx
from fastmcp import Client as McpClient
from loguru import logger

from xr_ai_models.openai_compat import OpenAICompatSTT, OpenAICompatTTS


# ── STT ───────────────────────────────────────────────────────────────────────

class SttClient:
    """OpenAI-compatible /v1/audio/transcriptions client.

    Thin wrapper around :class:`xr_ai_models.OpenAICompatSTT` that preserves
    the ``(base_url, timeout)`` constructor signature used by existing callers.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._stt = OpenAICompatSTT(base_url, timeout=timeout, client=client)

    @property
    def health_url(self) -> str:
        return self._stt.health_url

    async def transcribe(
        self, audio_data: bytes, sample_rate: int, channels: int = 1,
    ) -> str:
        # sample_rate triggers PCM→WAV conversion inside the SDK client.
        return await self._stt.transcribe(
            audio_data, sample_rate=sample_rate, channels=channels,
        )

    async def close(self) -> None:
        await self._stt.close()

    async def __aenter__(self) -> "SttClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ── TTS ───────────────────────────────────────────────────────────────────────

class TtsClient:
    """OpenAI-compatible /v1/audio/speech client.

    Thin wrapper around :class:`xr_ai_models.OpenAICompatTTS` that preserves
    the ``(base_url, timeout)`` constructor signature used by existing callers.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._tts = OpenAICompatTTS(base_url, timeout=timeout, client=client)

    @property
    def health_url(self) -> str:
        return self._tts.health_url

    async def synthesize(self, text: str) -> bytes:
        return await self._tts.synthesize(text)

    async def close(self) -> None:
        await self._tts.close()

    async def __aenter__(self) -> "TtsClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ── readiness probes ──────────────────────────────────────────────────────────

ProbeFn = Callable[[], Awaitable[bool]]


def http_probe(url: str, timeout: float = 3.0) -> ProbeFn:
    async def _probe() -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                return (await client.get(url)).is_success
        except httpx.HTTPError:
            return False
    return _probe


def mcp_probe(url: str) -> ProbeFn:
    """Probe an MCP server by calling list_tools() — the correct readiness check
    for pure-FastMCP servers that have no /health REST endpoint."""
    async def _probe() -> bool:
        try:
            async with McpClient(url) as mcp:
                await mcp.list_tools()
                return True
        except Exception:
            return False
    return _probe


async def wait_for_services(
    probes: dict[str, ProbeFn],
    *,
    poll_interval: float = 5.0,
) -> None:
    """Block until every named probe returns True."""
    pending = set(probes)
    while pending:
        for name in list(pending):
            if await probes[name]():
                logger.info("{} ready", name)
                pending.discard(name)
        if pending:
            logger.info("still waiting for: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(poll_interval)
