# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thin async clients for stt-server and tts-server, plus generic readiness probes.
"""
from __future__ import annotations

import asyncio
import io
import wave
from typing import Awaitable, Callable

import httpx
from fastmcp import Client as McpClient
from loguru import logger


# ── STT ───────────────────────────────────────────────────────────────────────

class SttClient:
    """OpenAI-compatible /v1/audio/transcriptions client."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        base = base_url.rstrip("/")
        self.health_url     = base + "/health"
        self.transcribe_url = base + "/v1/audio/transcriptions"
        self._client        = httpx.AsyncClient(timeout=timeout)

    async def transcribe(
        self, audio_data: bytes, sample_rate: int, channels: int = 1,
    ) -> str:
        wav_bytes = _pcm_to_wav(audio_data, sample_rate, channels)
        resp = await self._client.post(
            self.transcribe_url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json"},
        )
        if resp.is_error:
            logger.error("stt {}: {}", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def close(self) -> None:
        await self._client.aclose()


def _pcm_to_wav(pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ── TTS ───────────────────────────────────────────────────────────────────────

class TtsClient:
    """OpenAI-compatible /v1/audio/speech client."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        base = base_url.rstrip("/")
        self.health_url     = base + "/health"
        self.synthesize_url = base + "/v1/audio/speech"
        self._client        = httpx.AsyncClient(timeout=timeout)

    async def synthesize(self, text: str) -> bytes:
        resp = await self._client.post(
            self.synthesize_url,
            json={"input": text, "response_format": "wav"},
        )
        if resp.is_error:
            logger.error("tts {}: {}", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.content

    async def close(self) -> None:
        await self._client.aclose()


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
