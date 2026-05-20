# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible HTTP clients for the four service protocols.

Per-model quirks (reasoning field name, mandatory ``chat_template_kwargs``)
are absorbed by ``reasoning_field`` and ``default_extras`` on the
constructor; per-call quirks (``enable_thinking``, ``thinking_budget``)
fold into ``chat_template_kwargs`` on the wire.
"""
from __future__ import annotations

import base64
import io
import json
import os
import wave
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import httpx
from loguru import logger

from ._utils import merge_dicts
from .protocols import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    ContentPart,
    ImageInput,
    ImagePart,
    TextPart,
    ToolCall,
    ToolDef,
    VideoInput,
    VideoPart,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _msg_to_openai(msg: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role}
    if isinstance(msg.content, str):
        out["content"] = msg.content
    else:
        out["content"] = [_part_to_openai(p) for p in msg.content]
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    return out


def _part_to_openai(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {"type": "image_url", "image_url": {"url": part.url}}
    if isinstance(part, VideoPart):
        return {"type": "video_url", "video_url": {"url": part.url}}
    raise TypeError(f"unsupported content part: {type(part).__name__}")


def _parse_chat_response(data: dict[str, Any], reasoning_field: str | None) -> ChatResponse:
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""

    if reasoning_field:
        reasoning = msg.get(reasoning_field)
    else:
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")

    raw_tool_calls = msg.get("tool_calls")
    tool_calls: list[ToolCall] | None = None
    if raw_tool_calls:
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            for tc in raw_tool_calls
        ]

    return ChatResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )


_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff",       "image/jpeg"),
    (b"GIF87a",             "image/gif"),
    (b"GIF89a",             "image/gif"),
)


def _sniff_mime(data: bytes) -> str:
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _mime_from_suffix(p: Path) -> str:
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(p.suffix.lower(), "application/octet-stream")


def _to_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _normalize_image(image: ImageInput) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, Path):
        return _to_data_url(image.read_bytes(), _mime_from_suffix(image))
    if isinstance(image, (bytes, bytearray, memoryview)):
        b = bytes(image)
        return _to_data_url(b, _sniff_mime(b))
    raise TypeError(f"unsupported image input: {type(image).__name__}")


_VIDEO_SUFFIXES = {
    ".mp4":  "video/mp4",
    ".m4v":  "video/mp4",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".mkv":  "video/x-matroska",
}


def _sniff_video_mime(data: bytes) -> str:
    # ISO BMFF (mp4/mov/m4v): bytes 4..8 == "ftyp"; webm: EBML magic 1A 45 DF A3.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return "application/octet-stream"


def _video_mime_from_suffix(p: Path) -> str:
    return _VIDEO_SUFFIXES.get(p.suffix.lower(), "application/octet-stream")


def _normalize_video(video: VideoInput) -> str:
    if isinstance(video, str):
        return video
    if isinstance(video, Path):
        return _to_data_url(video.read_bytes(), _video_mime_from_suffix(video))
    if isinstance(video, (bytes, bytearray, memoryview)):
        b = bytes(video)
        return _to_data_url(b, _sniff_video_mime(b))
    raise TypeError(f"unsupported video input: {type(video).__name__}")


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── LLM ────────────────────────────────────────────────────────────────────


class OpenAICompatLLM:
    """OpenAI-compatible /v1/chat/completions client.

    Used directly for plain LLMs (Llama-Nemotron, Nemotron3-Nano,
    Nemotron-Omni) and indirectly via :class:`OpenAICompatVLM` for VLMs.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        capabilities: Capabilities | None = None,
        reasoning_field: str | None = None,
        default_extras: dict[str, Any] | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self._chat_url   = base + "/v1/chat/completions"
        self.health_url  = base + "/health"
        self._model      = model_name
        self.capabilities = capabilities or Capabilities()
        self._reasoning_field = reasoning_field
        self._default_extras  = default_extras or {}
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        self._client  = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None,
        max_tokens: int | None,
        temperature: float | None,
        enable_thinking: bool,
        thinking_budget: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_msg_to_openai(m) for m in messages],
        }
        if stream:
            payload["stream"] = True
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        per_call: dict[str, Any] = {}
        if enable_thinking or thinking_budget is not None:
            tpl: dict[str, Any] = {}
            if enable_thinking:
                tpl["enable_thinking"] = True
            if thinking_budget is not None:
                tpl["thinking_budget"] = thinking_budget
            per_call["chat_template_kwargs"] = tpl
        for k, v in merge_dicts(self._default_extras, per_call).items():
            payload[k] = v
        return payload

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        timeout: float | None = None,
    ) -> ChatResponse:
        payload = self._build_payload(
            messages,
            tools=tools, max_tokens=max_tokens, temperature=temperature,
            enable_thinking=enable_thinking, thinking_budget=thinking_budget,
            stream=False,
        )
        kwargs: dict[str, Any] = {"json": payload, "headers": self._headers()}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self._chat_url, **kwargs)
        if resp.is_error:
            logger.error("llm {} {}: {}", self._model, resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return _parse_chat_response(resp.json(), self._reasoning_field)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDef] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages,
            tools=tools, max_tokens=max_tokens, temperature=temperature,
            enable_thinking=enable_thinking, thinking_budget=thinking_budget,
            stream=True,
        )
        kwargs: dict[str, Any] = {"json": payload, "headers": self._headers()}
        if timeout is not None:
            kwargs["timeout"] = timeout
        async with self._client.stream("POST", self._chat_url, **kwargs) as resp:
            if resp.is_error:
                body = await resp.aread()
                logger.error("llm {} {}: {}", self._model, resp.status_code, body[:300])
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                content = delta.get("content")
                if content:
                    yield content

    async def health(self) -> bool:
        try:
            resp = await self._client.get(self.health_url, timeout=3.0)
            return resp.is_success
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatLLM":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ── VLM ────────────────────────────────────────────────────────────────────


class OpenAICompatVLM:
    """Vision LLM client — image-bearing chat completions, fronted as ``ask_image``."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        capabilities: Capabilities | None = None,
        default_extras: dict[str, Any] | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._llm = OpenAICompatLLM(
            base_url,
            model_name,
            capabilities=capabilities or Capabilities(vision=True),
            default_extras=default_extras,
            api_key_env=api_key_env,
            timeout=timeout,
            client=client,
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._llm.capabilities

    @property
    def health_url(self) -> str:
        return self._llm.health_url

    def _build_messages(
        self, image: ImageInput, question: str, system_prompt: str
    ) -> list[ChatMessage]:
        url = _normalize_image(image)
        msgs: list[ChatMessage] = []
        if system_prompt:
            msgs.append(ChatMessage(role="system", content=system_prompt))
        msgs.append(
            ChatMessage(
                role="user",
                content=[ImagePart(url=url), TextPart(text=question)],
            )
        )
        return msgs

    async def ask_image(
        self,
        image: ImageInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ChatResponse:
        return await self._llm.chat(
            self._build_messages(image, question, system_prompt),
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        )

    def _build_video_messages(
        self, video: VideoInput, question: str, system_prompt: str
    ) -> list[ChatMessage]:
        url = _normalize_video(video)
        msgs: list[ChatMessage] = []
        if system_prompt:
            msgs.append(ChatMessage(role="system", content=system_prompt))
        msgs.append(
            ChatMessage(
                role="user",
                content=[VideoPart(url=url), TextPart(text=question)],
            )
        )
        return msgs

    async def ask_video(
        self,
        video: VideoInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ChatResponse:
        if not self.capabilities.video:
            raise ValueError(
                "this VLM was constructed with capabilities.video=False; "
                "flip the preset / spec to declare video support before "
                "calling ask_video"
            )
        return await self._llm.chat(
            self._build_video_messages(video, question, system_prompt),
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        )

    async def stream(
        self,
        image: ImageInput,
        question: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._llm.stream(
            self._build_messages(image, question, system_prompt),
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        ):
            yield chunk

    async def health(self) -> bool:
        return await self._llm.health()

    async def close(self) -> None:
        await self._llm.close()

    async def __aenter__(self) -> "OpenAICompatVLM":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ── STT ────────────────────────────────────────────────────────────────────


class OpenAICompatSTT:
    """STT client — multipart POST to /v1/audio/transcriptions."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self._url       = base + "/v1/audio/transcriptions"
        self.health_url = base + "/health"
        self._api_key   = os.environ.get(api_key_env) if api_key_env else None
        self._client    = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        timeout: float | None = None,
    ) -> str:
        wav_bytes = _pcm_to_wav(audio, sample_rate, channels) if sample_rate is not None else audio
        kwargs: dict[str, Any] = {
            "files":   {"file": ("audio.wav", wav_bytes, "audio/wav")},
            "data":    {"response_format": "json"},
            "headers": self._headers(),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self._url, **kwargs)
        if resp.is_error:
            logger.error("stt {}: {}", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def health(self) -> bool:
        try:
            resp = await self._client.get(self.health_url, timeout=3.0)
            return resp.is_success
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatSTT":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ── TTS ────────────────────────────────────────────────────────────────────


class OpenAICompatTTS:
    """TTS client — JSON POST to /v1/audio/speech, returns raw audio bytes."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self._url       = base + "/v1/audio/speech"
        self.health_url = base + "/health"
        self._api_key   = os.environ.get(api_key_env) if api_key_env else None
        self._client    = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    async def synthesize(
        self,
        text: str,
        *,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes:
        kwargs: dict[str, Any] = {
            "json":    {"input": text, "response_format": response_format},
            "headers": self._headers(),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = await self._client.post(self._url, **kwargs)
        if resp.is_error:
            logger.error("tts {}: {}", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.content

    async def health(self) -> bool:
        try:
            resp = await self._client.get(self.health_url, timeout=3.0)
            return resp.is_success
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatTTS":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
