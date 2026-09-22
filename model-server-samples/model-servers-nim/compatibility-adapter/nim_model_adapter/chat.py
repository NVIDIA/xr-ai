# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate sample aliases and wire messages through the typed chat client."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from xr_ai_models import ChatMessage, OpenAICompatLLM, ToolDef

from .common import create_app


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


def _message(message: dict) -> ChatMessage:
    data = dict(message)
    content = data.get("content")
    if isinstance(content, list):
        data["content"] = [
            {"type": part["type"], "text": part["text"]} if part["type"] == "text"
            else {"type": part["type"], "url": part[part["type"]]["url"]}
            for part in content
        ]
    elif content is None:
        data["content"] = ""
    if data.get("tool_calls"):
        data["tool_calls"] = [{"id": call["id"], **call["function"]} for call in data["tool_calls"]]
    return TypeAdapter(ChatMessage).validate_python(data)


def build_app(config: dict, *, client_factory=None):
    def create_client(extras):
        defaults = dict(config.get("default_extras", {}))
        for key, value in extras.items():
            previous = defaults.get(key)
            defaults[key] = previous | value if isinstance(previous, dict) and isinstance(value, dict) else value
        return OpenAICompatLLM(
            config["base_url"], config["model"], default_extras=defaults,
            health_path=config.get("health_path", "/v1/health/ready"),
            reasoning_field=config.get("reasoning_field", "reasoning"), timeout=120,
        )

    factory = client_factory or create_client
    app = create_app(config, [factory({})])

    @app.post("/v1/chat/completions")
    async def complete(request: ChatRequest):
        if request.model not in (config["alias"], config["model"]):
            raise HTTPException(404, "unknown chat model")
        if request.stream and request.tools:
            raise HTTPException(400, "streaming function tools are unsupported; use stream: false")
        try:
            messages = [_message(message) for message in request.messages]
            tools = []
            for tool in request.tools or []:
                if tool.get("type") != "function":
                    raise ValueError("expected a function tool")
                function = {"description": "", "parameters": {"type": "object", "properties": {}},
                            **tool["function"]}
                tools.append(TypeAdapter(ToolDef).validate_python(function))
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise HTTPException(422, "invalid chat messages or function tools") from exc
        client = factory(request.model_extra or {})
        kwargs = {"tools": tools, "max_tokens": request.max_tokens, "temperature": request.temperature}
        if not request.stream:
            try:
                response = await client.chat(messages, **kwargs)
                result = dict(response.raw)
                result["model"] = request.model
                # Existing Nemotron clients read reasoning_content, while NIM
                # returns reasoning. Preserve the original provider fields too.
                if response.reasoning is not None:
                    result["choices"][0]["message"]["reasoning_content"] = response.reasoning
                return result
            finally:
                await client.close()

        chunks = client.stream(messages, **kwargs)
        try:
            # Detect upstream failures before sending HTTP 200 to the caller.
            first = await anext(chunks, None)
        except BaseException:
            await chunks.aclose()
            await client.close()
            raise
        stream_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())

        def event(content, finish_reason=None):
            return "data: " + json.dumps({
                "id": stream_id, "object": "chat.completion.chunk", "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": content} if content else {},
                             "finish_reason": finish_reason}],
            }) + "\n\n"

        async def stream():
            try:
                if first is not None:
                    yield event(first)
                async for text in chunks:
                    yield event(text)
                yield event(None, "stop")
                yield "data: [DONE]\n\n"
            finally:
                await chunks.aclose()
                await client.close()

        # The SDK streams visible text only. Function calls and reasoning are
        # available through non-streaming chat, as in existing agent clients.
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
