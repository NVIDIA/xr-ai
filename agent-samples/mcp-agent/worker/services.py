"""STT HTTP client, MCP client, and a readiness probe for both."""
from __future__ import annotations

import asyncio
import logging

import httpx
from fastmcp import Client as McpClient

log = logging.getLogger("mcp_agent.services")


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


async def _probe_http(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            return (await client.get(url)).is_success
    except httpx.HTTPError:
        return False


async def _probe_mcp(url: str) -> bool:
    """Pure-MCP readiness check — connect and list tools."""
    try:
        async with McpClient(url) as mcp:
            await mcp.list_tools()
            return True
    except Exception:
        return False


async def wait_for_services(stt_health_url: str, mcp_url: str) -> None:
    probes = {
        "STT": (_probe_http, stt_health_url),
        "MCP": (_probe_mcp,  mcp_url),
    }
    pending = set(probes)
    while pending:
        for name in list(pending):
            probe, url = probes[name]
            if await probe(url):
                log.info("%s ready", name)
                pending.discard(name)
        if pending:
            log.info("still waiting for: %s", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)
