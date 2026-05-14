# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub OpenAI-compatible HTTP server backed by ``httpx.MockTransport``.

Used by ``test_models_openai_compat.py`` to verify wire format without a
real server: the stub records every inbound request and lets the test set
canned responses for ``/v1/chat/completions`` (streaming and not),
``/v1/audio/transcriptions``, ``/v1/audio/speech``, and ``/health``.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


_DEFAULT_CHAT = {
    "choices": [{
        "message":      {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
    }],
}


class StubOpenAI:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies:   list[bytes] = []
        self._chat_status:    int = 200
        self._chat_response:  dict[str, Any] = dict(_DEFAULT_CHAT)
        self._stream_tokens:  list[str] = []
        self._transcribe_text: str = "stub-transcription"
        self._speech_bytes:    bytes = b"RIFF\x00\x00\x00\x00WAVEstub"
        self._health_status:  int = 200

    # ── configuration setters ──────────────────────────────────────────────

    def set_chat_message(
        self,
        *,
        content: str = "",
        reasoning: str | None = None,
        reasoning_field: str = "reasoning",
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg[reasoning_field] = reasoning
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self._chat_response = {
            "choices": [{"message": msg, "finish_reason": finish_reason}],
        }

    def set_chat_status(self, status: int) -> None:
        self._chat_status = status

    def set_stream_tokens(self, tokens: list[str]) -> None:
        self._stream_tokens = tokens

    def set_transcribe_text(self, text: str) -> None:
        self._transcribe_text = text

    def set_speech_bytes(self, data: bytes) -> None:
        self._speech_bytes = data

    def set_health_status(self, status: int) -> None:
        self._health_status = status

    # ── transport / client ─────────────────────────────────────────────────

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self, *, timeout: float = 5.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, timeout=timeout)

    # ── request inspection ─────────────────────────────────────────────────

    def last_json(self) -> dict[str, Any]:
        return json.loads(self.bodies[-1].decode())

    def last_request(self) -> httpx.Request:
        return self.requests[-1]

    # ── handler ────────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(request.content)
        path = request.url.path

        if path == "/health":
            return httpx.Response(self._health_status, text="ok")

        if path == "/v1/chat/completions":
            body = json.loads(request.content) if request.content else {}
            if body.get("stream"):
                sse_chunks = [
                    "data: " + json.dumps({
                        "choices": [{"delta": {"content": tok}}],
                    }) + "\n\n"
                    for tok in self._stream_tokens
                ]
                sse_chunks.append("data: [DONE]\n\n")
                return httpx.Response(
                    200,
                    content="".join(sse_chunks).encode(),
                    headers={"Content-Type": "text/event-stream"},
                )
            return httpx.Response(self._chat_status, json=self._chat_response)

        if path == "/v1/audio/transcriptions":
            return httpx.Response(200, json={"text": self._transcribe_text})

        if path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=self._speech_bytes,
                headers={"Content-Type": "audio/wav"},
            )

        return httpx.Response(404, text=f"unknown path: {path}")
