# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin async clients for the STT, VLM, TTS HTTP servers, and RAG MCP server."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx
from fastmcp import Client as McpClient
from loguru import logger


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
                logger.error("stt {}: {}", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.json().get("text", "")


class VlmClient:
    """OpenAI-compatible /v1/chat/completions client (SSE streaming) with vision."""

    def __init__(self, base_url: str, model_name: str = "vlm") -> None:
        base = base_url.rstrip("/")
        self.health_url = base + "/health"
        self.chat_url   = base + "/v1/chat/completions"
        self._model     = model_name

    async def stream(
        self,
        image_url: str | None,
        query: str,
        *,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """Yield text tokens from the VLM server via SSE.

        When *image_url* is ``None`` the call falls back to a text-only request
        so the agent can still answer questions when no camera frame is available.
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_url is None:
            messages.append({"role": "user", "content": query})
        else:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text",      "text": query},
            ]})

        payload = {
            "model":  self._model,
            "stream": True,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self.chat_url, json=payload) as resp:
                if resp.is_error:
                    logger.error("vlm-server {}", resp.status_code)
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
                logger.error("tts {}: {}", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.content


class RagClient:
    """MCP client for the rag-mcp-server's ``retrieve`` and ``list_documents`` tools."""

    def __init__(self, mcp_url: str) -> None:
        self._mcp_url = mcp_url.rstrip("/")
        # Exposed so wait_for_health can poll it — the MCP endpoint doubles
        # as a liveness indicator once FastMCP is serving.
        self.health_url = self._mcp_url

    async def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Call the rag-mcp ``retrieve`` tool; returns a list of {text, source, score}."""
        async with McpClient(self._mcp_url) as client:
            result = await client.call_tool("retrieve", {"query": query, "top_k": top_k})
        return result.data or []

    async def list_documents(self) -> list[str]:
        """Call the rag-mcp ``list_documents`` tool."""
        async with McpClient(self._mcp_url) as client:
            result = await client.call_tool("list_documents", {})
        return result.data or []


async def wait_for_health(services: dict[str, str]) -> None:
    """Poll each service URL until each one is alive.

    A service is "alive" once it returns any HTTP response — including 4xx
    like 406 (which the FastMCP /mcp endpoint returns for plain GETs).  We
    only retry on connect errors; the per-process ready file in the launcher
    already gates startup ordering.
    """
    pending = set(services)
    while pending:
        for name in list(pending):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    await client.get(services[name])
                logger.info("{} ready", name)
                pending.discard(name)
            except (httpx.ConnectError, httpx.ReadError):
                pass
        if pending:
            logger.info("still waiting for: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)
