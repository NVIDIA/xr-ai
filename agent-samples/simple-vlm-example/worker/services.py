# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin async clients for the STT, VLM, and TTS HTTP servers + readiness probe."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger("simple_vlm_example.services")


class SttClient:
    """OpenAI-compatible /v1/audio/transcriptions client."""

    def __init__(self, base_url: str) -> None:
        base = base_url.rstrip("/")
        self.health_url     = base + "/health"
        self.transcribe_url = base + "/v1/audio/transcriptions"

    async def transcribe(self, wav_bytes: bytes) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.transcribe_url,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"response_format": "json"},
            )
            if resp.is_error:
                log.error("stt %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.json().get("text", "")


class VlmClient:
    """OpenAI-compatible /v1/chat/completions client (SSE streaming)."""

    def __init__(self, base_url: str) -> None:
        base = base_url.rstrip("/")
        self.health_url = base + "/health"
        self.chat_url   = base + "/v1/chat/completions"

    async def stream(
        self,
        image_url: str,
        query: str,
        *,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """Yield text tokens from the VLM server via SSE."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text",      "text": query},
        ]})
        payload = {
            "model": "vlm",
            "stream": True,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self.chat_url, json=payload) as resp:
                if resp.is_error:
                    log.error("vlm-server %s", resp.status_code)
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        chunk   = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


class TtsClient:
    """OpenAI-compatible /v1/audio/speech client."""

    def __init__(self, base_url: str) -> None:
        base = base_url.rstrip("/")
        self.health_url     = base + "/health"
        self.synthesize_url = base + "/v1/audio/speech"

    async def synthesize(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.synthesize_url,
                json={"input": text, "response_format": "wav"},
            )
            if resp.is_error:
                log.error("tts %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.content


async def wait_for_health(services: dict[str, str]) -> None:
    """Poll each service's /health until all return 2xx.  Logs progress every 5 s."""
    pending = set(services)
    while pending:
        for name in list(pending):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    if (await client.get(services[name])).is_success:
                        log.info("%s ready", name)
                        pending.discard(name)
            except httpx.ConnectError:
                pass
        if pending:
            log.info("still waiting for: %s", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)
