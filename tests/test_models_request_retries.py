# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference retries are bounded and never replay an accepted response."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from xr_ai_models import (
    ChatMessage,
    OpenAICompatEmbedding,
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from xr_ai_models import _openai_compat as module


@pytest.fixture
def backoffs(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    # Replace this module's asyncio binding, not the process-wide sleep.
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=sleep))
    return delays


async def _invoke(kind, client):
    if kind == "llm":
        return (await OpenAICompatLLM("http://model", "test", client=client).chat(
            [ChatMessage(role="user", content="hello")], timeout=2,
        )).content
    if kind == "vlm":
        return (await OpenAICompatVLM("http://model", "test", client=client).ask_image(
            "https://example.com/image.jpg", "hello", timeout=2,
        )).content
    if kind == "stt":
        return await OpenAICompatSTT("http://model", client=client).transcribe(b"audio", timeout=2)
    if kind == "tts":
        return await OpenAICompatTTS("http://model", client=client).synthesize("hello", timeout=2)
    if kind == "embedding":
        return await OpenAICompatEmbedding("http://model", "test", client=client).embed(["hello"], timeout=2)
    if kind == "stream":
        llm = OpenAICompatLLM("http://model", "test", client=client)
        return "".join([part async for part in llm.stream([ChatMessage(role="user", content="hello")], timeout=2)])
    tts = module._PocketTTS("http://model", client=client)
    return b"".join([part.data async for part in tts.stream("hello", timeout=2)])


def _success(kind):
    if kind in {"llm", "vlm"}:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})
    if kind == "stt":
        return httpx.Response(200, json={"text": "hello"})
    if kind == "embedding":
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    if kind == "stream":
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n')
    return httpx.Response(200, content=b"\x01\x00", headers={"x-audio-sample-rate": "24000"})


@pytest.mark.parametrize("kind", ["llm", "vlm", "stt", "tts", "embedding", "stream", "pocket"])
@pytest.mark.parametrize("failure", [502, 503, 504, httpx.ConnectError, httpx.ConnectTimeout])
async def test_inference_retries_before_response_and_preserves_request(kind, failure, backoffs):
    requests = []
    failed_responses = []

    def handle(request):
        requests.append(request)
        if len(requests) < 3:
            if isinstance(failure, type):
                raise failure("not connected", request=request)
            response = httpx.Response(failure)
            failed_responses.append(response)
            return response
        return _success(kind)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await _invoke(kind, client)
    assert result in ("hello", b"\x01\x00", [[1.0]])
    assert len(requests) == 3
    assert all(r.method == "POST" and r.extensions["timeout"]["read"] == 2 for r in requests)
    assert len({str(r.url) for r in requests}) == 1
    if kind == "stt":
        assert all(b"audio" in r.content and b'filename="audio.wav"' in r.content for r in requests)
    else:
        assert all(json.loads(r.content) == json.loads(requests[0].content) for r in requests)
    assert all(r.is_closed for r in failed_responses)
    assert backoffs == [0.25, 0.5]


@pytest.mark.parametrize("failure, attempts", [
    (503, 3), (httpx.ConnectError, 3), (httpx.ConnectTimeout, 3),
    (400, 1), (401, 1), (403, 1), (404, 1), (429, 1), (500, 1),
    (httpx.ReadTimeout, 1), (httpx.WriteError, 1),
])
async def test_inference_failure_is_bounded_and_keeps_error_context(failure, attempts, backoffs):
    requests = []

    def handle(request):
        requests.append(request)
        if isinstance(failure, type):
            raise failure("unavailable", request=request)
        return httpx.Response(failure, text="model unavailable")

    error_type = failure if isinstance(failure, type) else httpx.HTTPStatusError
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(error_type) as caught:
            await _invoke("llm", client)
    assert str(caught.value.request.url) == "http://model/v1/chat/completions"
    if isinstance(failure, int):
        assert caught.value.response.status_code == failure
        assert caught.value.response.text == "model unavailable"
    assert len(requests) == attempts
    assert len(backoffs) == attempts - 1


@pytest.mark.parametrize("kind", ["stream", "pocket"])
@pytest.mark.parametrize("partial", [False, True])
async def test_stream_read_failure_never_replays_accepted_response(kind, partial, backoffs):
    calls = []
    closed = []

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            if partial:
                yield (b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                       if kind == "stream" else b"\x01\x00")
            raise httpx.ReadError("connection lost")

        async def aclose(self):
            closed.append(True)

    def handle(request):
        calls.append(request)
        return httpx.Response(200, stream=BrokenStream(), headers={"x-audio-sample-rate": "24000"})

    chunks = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        if kind == "stream":
            llm = OpenAICompatLLM("http://model", "test", client=client)
            iterator = llm.stream([ChatMessage(role="user", content="hello")])
        else:
            iterator = module._PocketTTS("http://model", client=client).stream("hello")
        with pytest.raises(httpx.ReadError, match="connection lost"):
            async for chunk in iterator:
                chunks.append(chunk)
    assert len(chunks) == int(partial)
    assert len(calls) == len(closed) == 1
    assert backoffs == []


async def test_cancellation_during_retry_backoff_stops_requests(monkeypatch):
    sleeping = asyncio.Event()
    requests = []
    responses = []

    async def sleep(_delay):
        sleeping.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=sleep))

    def handle(request):
        requests.append(request)
        response = httpx.Response(503)
        responses.append(response)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        task = asyncio.create_task(_invoke("llm", client))
        await asyncio.wait_for(sleeping.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(requests) == 1
    assert responses[0].is_closed
