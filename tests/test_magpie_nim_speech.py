# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real HTTP streaming against a controlled local Riva gRPC server."""
from __future__ import annotations

import asyncio
import io
import socket
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import grpc
import httpx
import pytest
import uvicorn
from magpie_nim_tts.speech import build_app
from riva.client.proto import riva_tts_pb2 as pb
from riva.client.proto import riva_tts_pb2_grpc as rpc
from xr_ai_models._riva_grpc import RivaTTS

FIRST = b"\x01\x02" * 80
LAST = b"\x03\x04" * 80
SILENCE = b"\x00\x00" * 13230  # 300 ms at 44,100 Hz.


class ControlledSpeech(rpc.RivaSpeechSynthesisServicer):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.calls = []

    def SynthesizeOnline(self, requests, context):
        # Qualified Riva clients use unary input; newer clients stream input.
        request = requests if isinstance(requests, pb.SynthesizeSpeechRequest) else next(requests)
        self.calls.append(request)
        self.started.set()
        try:
            if request.text == "empty":
                return
            if request.text == "fail-first":
                context.abort(grpc.StatusCode.UNAVAILABLE, "test failure")
            if request.text == "odd":
                yield pb.SynthesizeSpeechResponse(audio=b"\x01")
                return
            if request.text != "before-first":
                yield pb.SynthesizeSpeechResponse(audio=FIRST)
            if request.text != "immediate":
                while not self.release.wait(0.01):
                    if not context.is_active():
                        return
            if request.text == "fail-later":
                context.abort(grpc.StatusCode.INTERNAL, "test late failure")
            if context.is_active():
                yield pb.SynthesizeSpeechResponse(audio=LAST)
        finally:
            self.exited.set()


@asynccontextmanager
async def running_adapter(**config):
    backend = ControlledSpeech()
    pool = ThreadPoolExecutor(max_workers=2)
    grpc_server = grpc.server(pool)
    rpc.add_RivaSpeechSynthesisServicer_to_server(backend, grpc_server)
    grpc_port = grpc_server.add_insecure_port("127.0.0.1:0")
    grpc_server.start()
    app = build_app({"kind": "tts", "base_url": f"127.0.0.1:{grpc_port}",
                     "language": "en-US", "voice": "Magpie-Multilingual.EN-US.Aria", "sample_rate": 44100, **config})
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError("HTTP server exited during startup")
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5) as client:
            yield client, backend
    finally:
        backend.release.set()
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()
        grpc_server.stop(0).wait()
        pool.shutdown(wait=True)


def request(text="hold"):
    return {"input": text, "response_format": "pcm", "stream": True}


def test_first_pcm_reaches_http_client_before_grpc_finishes():
    async def run():
        async with running_adapter() as (client, backend):
            async with client.stream("POST", "/v1/audio/speech", json=request()) as response:
                assert response.status_code == 200
                assert response.headers["x-audio-sample-rate"] == "44100"
                assert response.headers["x-audio-channels"] == "1"
                assert response.headers["content-type"] == "audio/pcm"
                chunks = response.aiter_bytes()
                first = await asyncio.wait_for(anext(chunks), 2)
                assert first == FIRST
                assert not backend.release.is_set() and not backend.exited.is_set()
                backend.release.set()
                rest = b"".join([part async for part in chunks])
                assert first + rest == FIRST + LAST + SILENCE
            assert backend.calls[0].text == "hold"
            assert backend.calls[0].voice_name == "Magpie-Multilingual.EN-US.Aria"
            assert backend.calls[0].sample_rate_hz == 44100
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["before-first", "after-first", "queued"])
def test_disconnect_cancels_rpc_and_does_not_block_next_request(stage):
    async def run():
        async with running_adapter() as (client, backend):
            if stage == "before-first":
                pending = asyncio.create_task(client.post("/v1/audio/speech", json=request("before-first")))
                assert await asyncio.to_thread(backend.started.wait, 2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                async with client.stream("POST", "/v1/audio/speech", json=request()) as response:
                    chunks = response.aiter_bytes()
                    assert await anext(chunks) == FIRST
                    if stage == "queued":
                        pending = asyncio.create_task(client.post("/v1/audio/speech", json=request("queued")))
                        await asyncio.sleep(0.1)
                        assert len(backend.calls) == 1
                        pending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await pending
            assert await asyncio.to_thread(backend.exited.wait, 2)
            following = await client.post("/v1/audio/speech", json=request("immediate"))
            assert following.status_code == 200 and following.content == FIRST + LAST + SILENCE
            assert [call.text for call in backend.calls] == [
                "before-first" if stage == "before-first" else "hold", "immediate",
            ]
    asyncio.run(run())


def test_buffered_wav_waits_for_stream_and_preserves_audio():
    async def run():
        async with running_adapter() as (client, backend):
            async with client.stream("POST", "/v1/audio/speech", json=request()) as response:
                chunks = response.aiter_bytes()
                assert await anext(chunks) == FIRST
                wav = asyncio.create_task(client.post("/v1/audio/speech", json={"input": "immediate"}))
                await asyncio.sleep(0.1)
                assert not wav.done() and len(backend.calls) == 1
                backend.release.set()
                assert b"".join([part async for part in chunks]) == LAST + SILENCE
            result = await wav
            assert result.status_code == 200
            with wave.open(io.BytesIO(result.content)) as audio:
                assert audio.getframerate() == 44100
                assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
                assert audio.readframes(audio.getnframes()) == FIRST + LAST + SILENCE
    asyncio.run(run())


@pytest.mark.parametrize("text", ["fail-first", "odd"])
def test_failure_before_audio_returns_http_error(text):
    async def run():
        async with running_adapter() as (client, backend):
            response = await client.post("/v1/audio/speech", json=request(text))
            assert response.status_code == 502
            assert response.json() == {"detail": "Speech NIM request failed"}
            following = await client.post("/v1/audio/speech", json=request("immediate"))
            assert following.status_code == 200 and following.content == FIRST + LAST + SILENCE
    asyncio.run(run())


def test_failure_after_audio_aborts_http_stream():
    async def run():
        async with running_adapter() as (client, backend):
            async with client.stream("POST", "/v1/audio/speech", json=request("fail-later")) as response:
                assert response.status_code == 200
                chunks = response.aiter_bytes()
                assert await anext(chunks) == FIRST
                backend.release.set()
                with pytest.raises(httpx.RemoteProtocolError):
                    await anext(chunks)
            following = await client.post("/v1/audio/speech", json=request("immediate"))
            assert following.status_code == 200
    asyncio.run(run())


def test_timeout_before_audio_returns_504_and_releases_rpc(monkeypatch):
    original = RivaTTS._stream

    def short_timeout(self, text, *, timeout=None):
        return original(self, text, timeout=0.1 if text == "before-first" else 5)

    monkeypatch.setattr(RivaTTS, "_stream", short_timeout)

    async def run():
        async with running_adapter() as (client, backend):
            response = await client.post("/v1/audio/speech", json=request("before-first"))
            assert response.status_code == 504
            assert response.json() == {"detail": "NIM request timed out"}
            assert await asyncio.to_thread(backend.exited.wait, 2)
            following = await client.post("/v1/audio/speech", json=request("immediate"))
            assert following.status_code == 200 and following.content == FIRST + LAST + SILENCE
    asyncio.run(run())


@pytest.mark.parametrize("sample_rate,pause_ms,silent_samples", [
    (44100, 0, 0), (24000, 80, 1920), (22050, 150, 3308),
])
@pytest.mark.parametrize("stream,response_format", [(True, "pcm"), (False, "pcm"), (False, "wav")])
def test_configured_pause_matches_sample_rate_and_wav_length(sample_rate, pause_ms, silent_samples,
                                                          stream, response_format):
    async def run():
        async with running_adapter(sample_rate=sample_rate, post_synthesis_pause_ms=pause_ms) as (client, backend):
            response = await client.post("/v1/audio/speech", json={
                "input": "immediate", "stream": stream, "response_format": response_format,
            })
            assert response.status_code == 200
            expected = FIRST + LAST + b"\x00\x00" * silent_samples
            if response_format == "wav":
                with wave.open(io.BytesIO(response.content)) as wav:
                    assert wav.getframerate() == sample_rate
                    assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    assert wav.getnframes() == len(expected) // 2
                    assert wav.readframes(wav.getnframes()) == expected
            else:
                assert response.content == expected
            assert backend.calls[0].sample_rate_hz == sample_rate
    asyncio.run(run())


@pytest.mark.parametrize("stream,response_format", [(True, "pcm"), (False, "pcm"), (False, "wav")])
def test_empty_synthesis_does_not_produce_silence(stream, response_format):
    async def run():
        async with running_adapter() as (client, backend):
            response = await client.post("/v1/audio/speech", json={
                "input": "empty", "stream": stream, "response_format": response_format,
            })
            assert response.status_code == 200
            if response_format == "wav":
                with wave.open(io.BytesIO(response.content)) as wav:
                    assert wav.getnframes() == 0
            else:
                assert response.content == b""
            assert backend.exited.is_set()
    asyncio.run(run())


@pytest.mark.parametrize("pause_ms", [-1, 1.5, True, "150", None])
def test_invalid_pause_configuration_is_rejected(pause_ms):
    with pytest.raises(ValueError, match="post_synthesis_pause_ms must be a non-negative integer"):
        build_app({"kind": "tts", "post_synthesis_pause_ms": pause_ms}, backend=object())


@pytest.mark.parametrize("stream", [False, True])
def test_timeout_after_audio_cancels_rpc_and_allows_following_request(monkeypatch, stream):
    original = RivaTTS._stream

    def short_timeout(self, text, *, timeout=None):
        return original(self, text, timeout=0.1 if text == "hold" else 5)

    monkeypatch.setattr(RivaTTS, "_stream", short_timeout)

    async def run():
        async with running_adapter() as (client, backend):
            if stream:
                async with client.stream("POST", "/v1/audio/speech", json=request()) as response:
                    assert response.status_code == 200
                    chunks = response.aiter_bytes()
                    assert await anext(chunks) == FIRST
                    with pytest.raises(httpx.RemoteProtocolError):
                        await anext(chunks)
            else:
                response = await client.post("/v1/audio/speech", json=request() | {"stream": False})
                assert response.status_code == 504
            assert await asyncio.to_thread(backend.exited.wait, 2)
            following = await client.post("/v1/audio/speech", json=request("immediate"))
            assert following.status_code == 200 and following.content == FIRST + LAST + SILENCE
    asyncio.run(run())
