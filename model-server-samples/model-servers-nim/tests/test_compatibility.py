# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run unchanged consumer configurations through the real SDK and HTTP adapters."""
from __future__ import annotations

import asyncio
import io
import json
import wave
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest
from magpie_nim_tts.speech import build_app as tts_app
from nim_model_adapter.chat import build_app as chat_app
from nim_model_adapter.embedding import build_app as embedding_app
from nim_model_adapter.stt import build_app as stt_app
from xr_ai_models import (
    ChatMessage,
    ToolCall,
    ToolDef,
    load_models_config,
    make_embedding,
    make_llm,
    make_stt,
    make_tts,
    make_vlm,
)

BASE = Path(__file__).resolve().parents[1]
PCM = b"\x01\x00" * 160


def wav_bytes(pcm=PCM):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return output.getvalue()


class SpeechBackend:
    ready = True

    def __init__(self):
        self.calls = []

    async def health(self):
        return self.ready

    async def close(self):
        pass

    async def transcribe(self, audio, **kwargs):
        self.calls.append(audio)
        return "the model is ready"

    async def stream_pcm(self, text, **kwargs):
        self.calls.append((text, "pcm"))
        yield PCM

    async def synthesize(self, text, *, response_format, **kwargs):
        self.calls.append((text, response_format))
        return PCM if response_format == "pcm" else wav_bytes()


@pytest.fixture
def network(monkeypatch):
    real_client = httpx.AsyncClient
    calls = []
    state = {"status": 200}
    apps = {}
    speech = {kind: SpeechBackend() for kind in ("stt", "tts")}

    def native(request):
        if request.url.path == "/v1/health/ready":
            return httpx.Response(state["status"], json={"status": "ready"})
        body = json.loads(request.content)
        calls.append(body)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"detail": "backend error"})
        if request.url.path == "/v1/embeddings":
            assert body["model"].startswith("nvidia/embed-")
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [float(len(text))]} for i, text in enumerate(body["input"])
            ]})
        assert body["model"] in ("nvidia/omni", "nvidia/cosmos")
        if body.get("stream"):
            chunks = [{"choices": [{"delta": {"reasoning": "hidden"}}]}]
            chunks.extend({"choices": [{"delta": {"content": text}}]} for text in ("rea", "dy"))
            return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                                  + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        message = {"role": "assistant", "content": "ready", "reasoning": "because"}
        if body.get("tools"):
            message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {
                "name": "lookup", "arguments": '{"name":"test"}',
            }}]
        return httpx.Response(200, json={"model": body["model"], "choices": [{
            "index": 0, "message": message, "finish_reason": "tool_calls" if body.get("tools") else "stop",
        }], "usage": {"total_tokens": 12}})

    class Router(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.host.startswith("nim"):
                return native(request)
            return await httpx.ASGITransport(apps[request.url.port]).handle_async_request(request)

    router = Router()
    # Both sides use real SDK HTTP serialization. Only network transport is
    # replaced: consumer files and their model IDs, ports, and presets stay intact.
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=router, **kw))
    for kind, port in (("stt", 8103), ("tts", 8105)):
        speech_app = tts_app if kind == "tts" else stt_app
        apps[port] = speech_app({"kind": kind, "language": "en-US", "sample_rate": 44100,
                                "health_url": f"http://nim-{kind}/v1/health/ready"}, backend=speech[kind])
    for alias, model, port in (("llm", "omni", 8108), ("vlm", "cosmos", 8100)):
        apps[port] = chat_app({"kind": "chat", "alias": alias, "model": f"nvidia/{model}",
                              "base_url": f"http://nim-{alias}", "default_extras": {
                                  "chat_template_kwargs": {"enable_thinking": False},
                              }})
    apps[8109] = embedding_app("http://nim-embed", "nvidia/embed")
    return apps, speech, calls, state, real_client, router


@pytest.mark.parametrize("relative", [
    "agent-samples/simple-vlm-example/yaml/models.json",
    "model-server-samples/model-servers/yaml/models.default.json",
])
def test_existing_configs_work_unchanged(relative, network):
    apps, speech, calls, _, _, _ = network
    path = BASE.parents[1] / relative
    original = path.read_bytes()
    config = load_models_config(path)

    async def check():
        async with AsyncExitStack() as stack:
            for app in apps.values():
                await stack.enter_async_context(app.router.lifespan_context(app))
            clients = {}
            for name, factory in (("stt", make_stt), ("tts", make_tts), ("vlm", make_vlm),
                                  ("llm", make_llm), ("agent_llm", make_llm), ("embedding", make_embedding)):
                if name in config.entries:
                    client = clients[name] = factory(config, name)
                    stack.push_async_callback(client.close)
                    assert await client.health(), name
            assert await clients["tts"].synthesize("hello") == wav_bytes(PCM + b"\x00\x00" * 13230)
            chunks = [chunk async for chunk in clients["tts"].stream("stream hello")]
            assert b"".join(chunk.data for chunk in chunks) == PCM + b"\x00\x00" * 13230
            assert all(chunk.sample_rate == 44100 and chunk.channels == 1 for chunk in chunks)
            assert speech["tts"].calls == [("hello", "pcm"), ("stream hello", "pcm")]
            assert await clients["stt"].transcribe(wav_bytes()) == "the model is ready"
            assert speech["stt"].calls == [wav_bytes()]
            response = await clients["vlm"].ask_image("data:image/png;base64,aGVsbG8=", "describe", max_tokens=32)
            assert response.content == "ready"
            image = calls[-1]["messages"][0]["content"][0]
            assert image == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            text = "".join([chunk async for chunk in clients["vlm"].stream("data:image/png;base64,aGVsbG8=", "hello")])
            assert text == "ready"
            for name in ("llm", "agent_llm"):
                if name not in clients:
                    continue
                messages = [ChatMessage("assistant", "", tool_calls=[ToolCall("prev", "lookup", "{}")]),
                            ChatMessage("tool", "found", tool_call_id="prev"), ChatMessage("user", "find test")]
                response = await clients[name].chat(messages, tools=[ToolDef("lookup", "Find a name", {
                    "type": "object", "properties": {"name": {"type": "string"}},
                })], enable_thinking=True, thinking_budget=64, max_tokens=128, temperature=0.1)
                assert response.tool_calls == [ToolCall("call-1", "lookup", '{"name":"test"}')]
                assert response.reasoning == "because"
                assert response.raw["model"] == "llm"
                assert response.raw["usage"] == {"total_tokens": 12}
                upstream = calls[-1]
                assert upstream["messages"][0]["content"] is None
                assert upstream["messages"][1]["tool_call_id"] == "prev"
                assert upstream["max_tokens"] == 128 and upstream["temperature"] == 0.1
                assert upstream["chat_template_kwargs"] == {"enable_thinking": True, "thinking_budget": 64}
                assert "".join([text async for text in clients[name].stream([ChatMessage("user", "hi")])]) == "ready"
            if "embedding" in clients:
                assert await clients["embedding"].embed(["passage: abcd", "query: hi"]) == [[4.0], [2.0]]
    asyncio.run(check())
    assert path.read_bytes() == original


def test_backend_errors_and_invalid_requests_are_not_reported_as_ready(network):
    apps, speech, _, state, real_client, router = network

    async def check():
        async with AsyncExitStack() as stack:
            for app in apps.values():
                await stack.enter_async_context(app.router.lifespan_context(app))
            client = await stack.enter_async_context(real_client(transport=router))
            for port in (8103, 8105, 8108, 8100, 8109):
                assert (await client.get(f"http://localhost:{port}/health")).status_code == 200
            # Previously exported NIM profiles use the native model ID and
            # readiness path. They remain valid alongside the original aliases.
            assert (await client.get("http://localhost:8108/v1/health/ready")).status_code == 200
            native_id = await client.post("http://localhost:8108/v1/chat/completions", json={
                "model": "nvidia/omni", "messages": [{"role": "user", "content": "hi"}],
            })
            assert native_id.status_code == 200 and native_id.json()["model"] == "nvidia/omni"
            state["status"] = 503
            assert (await client.get("http://localhost:8108/v1/health/ready")).status_code == 503
            for port in (8103, 8105, 8108, 8100, 8109):
                assert (await client.get(f"http://localhost:{port}/health")).status_code == 503
            for stream in (True, False):
                response = await client.post("http://localhost:8108/v1/chat/completions", json={
                    "model": "llm", "messages": [{"role": "user", "content": "hi"}], "stream": stream,
                })
                assert response.status_code == 503
            state["status"] = 200
            speech["tts"].ready = False
            assert (await client.get("http://localhost:8105/health")).status_code == 503
            invalid_audio = await client.post("http://localhost:8103/v1/audio/transcriptions",
                                              files={"file": ("bad.wav", b"not a WAV")})
            assert invalid_audio.status_code == 400
            invalid_stream = await client.post("http://localhost:8105/v1/audio/speech", json={
                "input": "hello", "response_format": "wav", "stream": True,
            })
            assert invalid_stream.status_code == 400
            for body, code in [({"model": "wrong", "messages": []}, 404), ({"model": "llm", "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {}}]},
            ]}, 422)]:
                assert (await client.post("http://localhost:8108/v1/chat/completions", json=body)).status_code == code
    asyncio.run(check())


def test_closing_chat_stream_releases_upstream_generator_and_client():
    from nim_model_adapter.chat import ChatRequest

    class Client:
        closed = False
        stream_closed = False

        async def close(self):
            self.closed = True

        async def stream(self, *args, **kwargs):
            try:
                yield "first"
                await asyncio.Future()
            finally:
                self.stream_closed = True

    async def check():
        clients = []

        def factory(extras):
            client = Client()
            clients.append(client)
            return client

        app = chat_app({"alias": "llm", "model": "nvidia/model"}, client_factory=factory)
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/chat/completions")
        async with app.router.lifespan_context(app):
            response = await endpoint(ChatRequest(model="llm", messages=[{"role": "user", "content": "hi"}],
                                                  stream=True))
            assert "first" in await anext(response.body_iterator)
            await response.body_iterator.aclose()
            assert clients[-1].closed and clients[-1].stream_closed
        assert all(client.closed for client in clients)
    asyncio.run(check())
