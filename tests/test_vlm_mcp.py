# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vlm-mcp's FastMCP wrapper around vlm-server.

vlm-mcp is a pure HTTP wrapper — no GPU, no hub IPC. These tests exercise the
``ask_image`` MCP tool and the ``_make_vlm_from_cfg`` factory.

Wire-shape contract (data URL encoding, prompt forwarding, model field,
``enable_thinking`` knob) and the response-relay path are covered using both
the ``StubOpenAI`` httpx mock-transport (for wire assertions without a running
server) and an in-process aiohttp server (for the legacy back-compat path and
the unreachable-server error path).
"""
from __future__ import annotations

import base64
import socket
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from xr_ai_models.openai_compat import OpenAICompatVLM

from vlm_mcp_server.__main__ import (
    _load_jpeg_data_url,
    _make_vlm_from_cfg,
    build_mcp,
)
from _stub_openai import StubOpenAI

pytestmark = pytest.mark.asyncio


# ── helpers ────────────────────────────────────────────────────────────────────

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
            "choices": [{"message": {"content": self.answer}, "finish_reason": "stop"}],
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


# ── fixtures ──────────────────────────────────────────────────────────────────

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


def _stub_vlm(stub: StubOpenAI, *, enable_thinking: bool = False) -> OpenAICompatVLM:
    """Build an OpenAICompatVLM wired to *stub* — no real server needed."""
    extras: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return OpenAICompatVLM(
        base_url="http://stub",
        model_name="vlm",
        default_extras=extras,
        timeout=5.0,
        client=stub.client(timeout=5.0),
    )


# ── image helper tests ─────────────────────────────────────────────────────────

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


# ── wire-shape golden tests (StubOpenAI) ──────────────────────────────────────

async def test_ask_image_relays_response_and_request_shape(png_path: Path):
    """Wire shape with enable_thinking=False (default / cosmos_vlm preset default)."""
    stub = StubOpenAI()
    stub.set_chat_message(content="a cat sitting on a mat")
    async with _stub_vlm(stub, enable_thinking=False) as vlm:
        mcp = build_mcp(vlm)

        result = await mcp.call_tool(
            "ask_image",
            {"question": "what is in this image?", "image_path": str(png_path)},
        )
        text = result.structured_content["result"]
        assert text == "a cat sitting on a mat"

        payload = stub.last_json()
        assert payload["model"] == "vlm"
        # enable_thinking=False must surface as chat_template_kwargs on the wire.
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

        msgs = payload["messages"]
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        parts = msgs[0]["content"]
        image_part = next(p for p in parts if p["type"] == "image_url")
        text_part  = next(p for p in parts if p["type"] == "text")
        assert text_part["text"] == "what is in this image?"
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_ask_image_enable_thinking_sets_template_kwarg_true(png_path: Path):
    """When enable_thinking=True, chat_template_kwargs.enable_thinking is True on the wire."""
    stub = StubOpenAI()
    stub.set_chat_message(content="the answer is yes")
    async with _stub_vlm(stub, enable_thinking=True) as vlm:
        mcp = build_mcp(vlm)
        await mcp.call_tool(
            "ask_image",
            {"question": "describe", "image_path": str(png_path)},
        )
        payload = stub.last_json()
        # enable_thinking=True must be explicit on the wire.
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}


async def test_ask_image_strips_think_block(png_path: Path):
    """<think>…</think> blocks are stripped from the VLM response."""
    stub = StubOpenAI()
    stub.set_chat_message(content="<think>let me look</think>\n  the answer is yes  ")
    async with _stub_vlm(stub) as vlm:
        mcp = build_mcp(vlm)
        result = await mcp.call_tool(
            "ask_image",
            {"question": "q?", "image_path": str(png_path)},
        )
    assert result.structured_content["result"] == "the answer is yes"


# ── error / edge-case tests ───────────────────────────────────────────────────

async def test_ask_image_missing_path_returns_error_string(png_path: Path):
    """Missing image_path is a user error — must not raise, returns guidance."""
    stub = StubOpenAI()
    async with _stub_vlm(stub) as vlm:
        mcp = build_mcp(vlm)

        empty = await mcp.call_tool("ask_image", {"question": "q", "image_path": ""})
        assert empty.structured_content["result"].startswith("ask_image: image_path is empty")

        missing = await mcp.call_tool(
            "ask_image",
            {"question": "q", "image_path": "/nonexistent/path/should/not/exist.png"},
        )
        assert "file not found" in missing.structured_content["result"]


async def test_ask_image_http_error_returns_error_string(png_path: Path):
    """When vlm-server returns 5xx, the tool returns an error string."""
    stub = StubOpenAI()
    stub.set_chat_status(500)
    async with _stub_vlm(stub) as vlm:
        mcp = build_mcp(vlm)
        result = await mcp.call_tool(
            "ask_image",
            {"question": "q", "image_path": str(png_path)},
        )
    assert result.structured_content["result"].startswith("ask_image: vlm-server request failed")


# ── legacy back-compat path (vlm_server: URL key) ────────────────────────────

async def test_make_vlm_from_cfg_legacy_path(mock_vlm, png_path: Path):
    """Legacy vlm_server: URL key synthesises a working VLMService."""
    server, base_url = mock_vlm
    server.answer = "legacy answer"

    vlm, timeout = _make_vlm_from_cfg({
        "vlm_server":            base_url,
        "vlm_request_timeout_s": 5.0,
        "enable_thinking":       False,
    })
    assert timeout == 5.0
    mcp = build_mcp(vlm)
    try:
        result = await mcp.call_tool(
            "ask_image",
            {"question": "legacy q", "image_path": str(png_path)},
        )
    finally:
        await vlm.close()

    assert result.structured_content["result"] == "legacy answer"
    assert len(server.requests) == 1
    payload = server.requests[0]
    assert payload["model"] == "vlm"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


async def test_make_vlm_from_cfg_new_models_block(mock_vlm, png_path: Path):
    """New models: block with preset:cosmos_vlm produces correct wire shape."""
    server, base_url = mock_vlm
    server.answer = "cosmos answer"

    vlm, _ = _make_vlm_from_cfg({
        "models": {
            "vlm": {
                "kind":     "preset:cosmos_vlm",
                "base_url": base_url,
            },
        },
        "vlm_request_timeout_s": 5.0,
        "enable_thinking":       False,
    })
    mcp = build_mcp(vlm)
    try:
        result = await mcp.call_tool(
            "ask_image",
            {"question": "cosmos q", "image_path": str(png_path)},
        )
    finally:
        await vlm.close()

    assert result.structured_content["result"] == "cosmos answer"
    payload = server.requests[0]
    assert payload["model"] == "vlm"
    # cosmos_vlm preset sets enable_thinking=False by default.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


async def test_make_vlm_from_cfg_missing_required_keys_raises():
    """Neither models: nor vlm_server: present → ValueError."""
    with pytest.raises(ValueError, match="must specify either"):
        _make_vlm_from_cfg({})


# ── stub-server wire-trace golden ─────────────────────────────────────────────

async def test_wire_trace_golden_matches_pre_migration_shape(png_path: Path):
    """Assert the SDK produces the same JSON body shape as the pre-migration VlmClient.

    Pre-migration golden (captured from VlmClient.ask with enable_thinking=False):
      - model: "vlm"
      - messages: [{"role": "user", "content": [image_url_part, text_part]}]
      - chat_template_kwargs: {"enable_thinking": false}

    The cosmos_vlm preset's default_extras carry the same chat_template_kwargs,
    so the wire body is identical.
    """
    stub = StubOpenAI()
    stub.set_chat_message(content="answer")
    async with _stub_vlm(stub, enable_thinking=False) as vlm:
        mcp = build_mcp(vlm)
        await mcp.call_tool(
            "ask_image",
            {"question": "test question", "image_path": str(png_path)},
        )

    body = stub.last_json()

    # Model field — must be "vlm" (matches pre-migration VlmClient hard-coded string).
    assert body["model"] == "vlm"

    # chat_template_kwargs — present and has enable_thinking=false (matches pre-migration).
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    # Message structure — single user turn with image_url first, text second.
    assert len(body["messages"]) == 1
    msg = body["messages"][0]
    assert msg["role"] == "user"
    parts = msg["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "test question"
