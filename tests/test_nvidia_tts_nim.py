# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Speech TTS NIM HTTP adapter coverage."""

from __future__ import annotations

import httpx
from xr_ai_models import NvidiaTTSNIM


class _ChunkedBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


async def test_magpie_nim_streams_raw_pcm_in_playable_frames() -> None:
    requests: list[httpx.Request] = []
    frame_bytes = 22050 * 20 // 1000 * 2

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        requests.append(request)
        return httpx.Response(
            200,
            stream=_ChunkedBody([
                b"a" * 100,
                b"b" * (frame_bytes - 100),
                b"c" * 11,
            ]),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = NvidiaTTSNIM(
        "http://nim:9000",
        language_code="en-US",
        voice="Magpie-Multilingual.EN-US.Mia",
        client=http,
    )

    chunks = [chunk async for chunk in tts.stream_synthesize("Hello.")]

    assert [len(chunk.data) for chunk in chunks] == [frame_bytes, 10]
    assert all(chunk.sample_rate == 22050 for chunk in chunks)
    assert all(chunk.channels == 1 for chunk in chunks)
    assert requests[0].url.path == "/v1/audio/synthesize_online"
    body = requests[0].content.decode(errors="replace")
    assert 'name="text"' in body and "Hello." in body
    assert 'name="language"' in body and "en-US" in body
    assert 'name="voice"' in body and "Magpie-Multilingual.EN-US.Mia" in body
    assert 'name="sample_rate_hz"' in body and "22050" in body
    assert 'name="encoding"' in body and "LINEAR_PCM" in body
    await http.aclose()


async def test_magpie_nim_offline_fallback_and_health() -> None:
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        paths.append(request.url.path)
        if request.url.path == "/v1/health/ready":
            return httpx.Response(200)
        return httpx.Response(200, content=b"RIFF-test-WAVE")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = NvidiaTTSNIM(
        "http://nim:9000/",
        language_code="en-US",
        voice="Magpie-Multilingual.EN-US.Mia",
        client=http,
    )

    assert await tts.health() is True
    assert await tts.synthesize("Fallback.") == b"RIFF-test-WAVE"
    assert paths == ["/v1/health/ready", "/v1/audio/synthesize"]
    await http.aclose()


async def test_magpie_nim_health_can_be_disabled() -> None:
    async def unexpected(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("health_check=False must not perform HTTP")

    http = httpx.AsyncClient(transport=httpx.MockTransport(unexpected))
    tts = NvidiaTTSNIM(
        "http://nim:9000",
        language_code="en-US",
        voice="Magpie-Multilingual.EN-US.Mia",
        health_check=False,
        client=http,
    )

    assert await tts.health() is True
    await http.aclose()
