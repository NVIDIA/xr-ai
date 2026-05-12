# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vlm-mcp's FastMCP wrapper around vlm-server.

vlm-mcp is a pure HTTP wrapper — no GPU, no hub IPC. These tests stand up an
in-process aiohttp mock for the upstream vlm-server `/v1/chat/completions`
endpoint, wire vlm-mcp's ``VlmClient`` at it, and exercise the ``ask_image``
MCP tool against a synthetic PNG.

Both the wire-shape contract (data URL encoding, prompt forwarding, model
field, ``enable_thinking`` knob) and the response-relay path are covered.
"""
from __future__ import annotations

import base64
import socket
import struct
import zlib
from pathlib import Path

import pytest
from aiohttp import web

from vlm_mcp_server.__main__ import VlmClient, _load_jpeg_data_url, build_mcp

pytestmark = pytest.mark.asyncio


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tiny_png_bytes(size: int = 8) -> bytes:
    """Return a minimal valid grayscale PNG (no external deps)."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    raw  = b"".join(b"\x00" + bytes([(i * 8) & 0xFF] * size) for i in range(size))
    idat = zlib.compress(raw, 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


class _MockVlmServer:
    """Tiny aiohttp app standing in for vlm-server's chat-completions route."""

    def __init__(self, answer: str = "a cat sitting on a mat") -> None:
        self.answer = answer
        self.requests: list[dict] = []
        self.runner: web.AppRunner | None = None
        self.port: int = 0

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        return web.json_response({
            "choices": [{"message": {"content": self.answer}}],
        })

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.port = _pick_free_port()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await site.start()
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()


@pytest.fixture
async def mock_vlm():
    server = _MockVlmServer()
    base_url = await server.start()
    try:
        yield server, base_url
    finally:
        await server.stop()


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    p = tmp_path / "frame.png"
    p.write_bytes(_tiny_png_bytes())
    return p


async def test_load_jpeg_data_url_emits_data_url(png_path: Path):
    url = _load_jpeg_data_url(str(png_path))
    assert url.startswith("data:image/jpeg;base64,")
    # The payload must round-trip through base64 cleanly.
    head, _, payload = url.partition(",")
    assert head == "data:image/jpeg;base64"
    raw = base64.b64decode(payload)
    # JPEG SOI / EOI markers — proves PIL re-encoded as JPEG, not just relayed PNG.
    assert raw[:2] == b"\xff\xd8"
    assert raw[-2:] == b"\xff\xd9"


async def test_ask_image_relays_response_and_request_shape(mock_vlm, png_path: Path):
    server, base_url = mock_vlm
    vlm = VlmClient(base_url, timeout=5.0, enable_thinking=False)
    try:
        mcp = build_mcp(vlm)
        result = await mcp.call_tool(
            "ask_image",
            {"question": "what is in this image?", "image_path": str(png_path)},
        )
    finally:
        await vlm.close()

    # FastMCP returns the str directly under structured_content['result'].
    text = result.structured_content["result"]
    assert text == "a cat sitting on a mat"

    assert len(server.requests) == 1
    payload = server.requests[0]
    assert payload["model"] == "vlm"
    # enable_thinking=False must surface as chat_template_kwargs to suppress <think>.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    msgs = payload["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    parts = msgs[0]["content"]
    image_part = next(p for p in parts if p["type"] == "image_url")
    text_part  = next(p for p in parts if p["type"] == "text")
    assert text_part["text"] == "what is in this image?"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_ask_image_enable_thinking_omits_template_kwarg(mock_vlm, png_path: Path):
    """When enable_thinking=True, the chat_template_kwargs key must NOT appear."""
    server, base_url = mock_vlm
    vlm = VlmClient(base_url, timeout=5.0, enable_thinking=True)
    try:
        mcp = build_mcp(vlm)
        await mcp.call_tool(
            "ask_image",
            {"question": "describe", "image_path": str(png_path)},
        )
    finally:
        await vlm.close()
    assert "chat_template_kwargs" not in server.requests[0]


async def test_ask_image_strips_think_block(mock_vlm, png_path: Path):
    server, base_url = mock_vlm
    server.answer = "<think>let me look</think>\n  the answer is yes  "
    vlm = VlmClient(base_url, timeout=5.0)
    try:
        mcp = build_mcp(vlm)
        result = await mcp.call_tool(
            "ask_image",
            {"question": "q?", "image_path": str(png_path)},
        )
    finally:
        await vlm.close()
    assert result.structured_content["result"] == "the answer is yes"


async def test_ask_image_missing_path_returns_error_string():
    """Missing image_path is a user error — must not raise, returns guidance."""
    vlm = VlmClient("http://127.0.0.1:1", timeout=1.0)
    try:
        mcp = build_mcp(vlm)
        # Empty string → guidance error.
        empty = await mcp.call_tool("ask_image", {"question": "q", "image_path": ""})
        assert empty.structured_content["result"].startswith("ask_image: image_path is empty")
        # Non-existent file → file-not-found error.
        missing = await mcp.call_tool(
            "ask_image",
            {"question": "q", "image_path": "/nonexistent/path/should/not/exist.png"},
        )
        assert "file not found" in missing.structured_content["result"]
    finally:
        await vlm.close()


async def test_ask_image_http_error_returns_error_string(png_path: Path):
    """When vlm-server is unreachable, the tool returns an error string —
    it must never raise into the agent's tool-call loop."""
    # Point the client at a closed port.
    vlm = VlmClient(f"http://127.0.0.1:{_pick_free_port()}", timeout=1.0)
    try:
        mcp = build_mcp(vlm)
        result = await mcp.call_tool(
            "ask_image",
            {"question": "q", "image_path": str(png_path)},
        )
    finally:
        await vlm.close()
    assert result.structured_content["result"].startswith("ask_image: vlm-server request failed")
