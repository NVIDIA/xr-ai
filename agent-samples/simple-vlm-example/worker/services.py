# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin async clients for the STT, VLM, TTS, and pose-mcp servers + readiness probe."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
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
    """OpenAI-compatible /v1/chat/completions client (SSE streaming)."""

    def __init__(self, base_url: str, model_name: str = "vlm") -> None:
        base = base_url.rstrip("/")
        self.health_url = base + "/health"
        self.chat_url   = base + "/v1/chat/completions"
        self._model     = model_name

    async def collect(
        self,
        image_url: str,
        query: str,
        *,
        system_prompt: str = "",
        max_chars:     int = 2000,
    ) -> str:
        """Non-streaming wrapper around `stream` — concatenates SSE chunks
        into a single response string.  Cheap convenience for short
        structured outputs (e.g. JSON object lists) where we don't want
        the caller to manage a token loop."""
        buf: list[str] = []
        async for tok in self.stream(image_url, query, system_prompt=system_prompt):
            buf.append(tok)
            if sum(len(s) for s in buf) > max_chars:
                break
        return "".join(buf).strip()

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
            "model": self._model,
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


class SpaceClient:
    """FastMCP client for space-mcp's topological place-memory tools.

    Long-lived MCP connection; reopens transparently on a dropped call.
    Set ``url`` to ``None`` to disable the space path entirely.
    """

    def __init__(self, url: str) -> None:
        self._url    = url.rstrip("/")
        self._client: Any = None
        self._lock   = asyncio.Lock()

    @property
    def health_url(self) -> str:
        return self._url

    async def _ensure_open(self) -> None:
        if self._client is not None:
            return
        from fastmcp import Client
        self._client = Client(self._url)
        await self._client.__aenter__()

    async def _aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None

    async def _call(self, tool: str, args: dict) -> dict:
        async with self._lock:
            try:
                await self._ensure_open()
                result = await self._client.call_tool(tool, args)
            except Exception:
                await self._aclose()
                raise
        if hasattr(result, "data") and result.data is not None:
            return result.data
        try:
            return json.loads(result.content[0].text)
        except Exception:
            return {"error": "unparseable space-mcp response"}

    async def process_frame(self, image_path: str, timestamp_us: int = 0) -> dict:
        return await self._call("process_frame", {
            "image_path": image_path, "timestamp_us": timestamp_us,
        })

    async def remember_objects(
        self, region_id: int, names: list[str], timestamp_us: int = 0,
    ) -> dict:
        return await self._call("remember_objects", {
            "region_id": int(region_id),
            "names":     list(names),
            "timestamp_us": int(timestamp_us),
        })

    async def close(self) -> None:
        async with self._lock:
            await self._aclose()


async def wait_for_space_mcp(client: "SpaceClient | None") -> None:
    """Poll until ``list_regions`` answers — proves FastMCP is up."""
    if client is None:
        return
    while True:
        try:
            await client._ensure_open()
            await client._client.call_tool("list_regions", {})
            logger.info("SPACE ready")
            return
        except Exception as exc:
            logger.info("still waiting for SPACE: {}", exc.__class__.__name__)
            await client._aclose()
            await asyncio.sleep(5.0)


class PoseClient:
    """FastMCP client for pose-mcp's ``estimate_pose`` tool.

    Holds a long-lived MCP connection so each call is a single round trip; if
    the connection drops the next call re-opens transparently.  Set ``url`` to
    ``None`` to disable the pose path entirely (the worker treats the absence
    of a client as a no-op feature flag).
    """

    def __init__(self, url: str) -> None:
        self._url     = url.rstrip("/")
        self._client: Any = None
        self._lock    = asyncio.Lock()

    @property
    def health_url(self) -> str:
        # FastMCP's StreamableHTTP transport exposes /mcp; a HEAD/GET that gets
        # a 4xx is still proof the server is alive, so wait_for_health's
        # `is_success` would miss it — use the readiness flag instead.
        return self._url

    async def _ensure_open(self) -> None:
        if self._client is not None:
            return
        from fastmcp import Client
        self._client = Client(self._url)
        await self._client.__aenter__()

    async def _aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None

    async def estimate_pose(self, image_path: str, timestamp_us: int = 0) -> dict:
        async with self._lock:
            try:
                await self._ensure_open()
                result = await self._client.call_tool(
                    "estimate_pose",
                    {"image_path": image_path, "timestamp_us": timestamp_us},
                )
            except Exception:
                # Drop the broken connection so the next call retries fresh.
                await self._aclose()
                raise
        # FastMCP returns a `CallToolResult` whose `.data` holds the parsed
        # JSON payload; older releases expose it under `.content[0].text`.
        if hasattr(result, "data") and result.data is not None:
            return result.data
        try:
            return json.loads(result.content[0].text)
        except Exception:
            return {"error": "unparseable pose-mcp response"}

    async def close(self) -> None:
        async with self._lock:
            await self._aclose()


async def wait_for_pose_mcp(client: "PoseClient") -> None:
    """Poll pose-mcp until ``get_map_stats`` answers — proves the FastMCP
    handler is up.  Logs progress every 5 s.  No-op when ``client`` is None."""
    if client is None:
        return
    while True:
        try:
            await client._ensure_open()
            await client._client.call_tool("get_map_stats", {})
            logger.info("POSE ready")
            return
        except Exception as exc:
            logger.info("still waiting for POSE: {}", exc.__class__.__name__)
            await client._aclose()
            await asyncio.sleep(5.0)


async def wait_for_health(services: dict[str, str]) -> None:
    """Poll each service's /health until all return 2xx.  Logs progress every 5 s."""
    pending = set(services)
    while pending:
        for name in list(pending):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    if (await client.get(services[name])).is_success:
                        logger.info("{} ready", name)
                        pending.discard(name)
            except httpx.ConnectError:
                pass
        if pending:
            logger.info("still waiting for: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)
