# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-trace tests for xr_ai_pipecat.SttClient and TtsClient.

Verifies that the pipecat wrappers produce exactly the same HTTP wire format as
the pre-migration hand-rolled httpx clients did — multipart form POST for STT
and JSON POST for TTS — by injecting an httpx.MockTransport stub.
"""
from __future__ import annotations

import io
import json
import wave

from _stub_openai import StubOpenAI
from xr_ai_pipecat.services import SttClient, TtsClient


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pcm(n_frames: int = 160) -> bytes:
    """16-bit silence at 16 kHz — smallest valid PCM block."""
    return b"\x00\x00" * n_frames


def _parse_wav_header(data: bytes) -> dict:
    """Return sample_rate, channels, and sample_width from a WAV header."""
    buf = io.BytesIO(data)
    with wave.open(buf, "rb") as wf:
        return {
            "channels":     wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate":  wf.getframerate(),
            "n_frames":     wf.getnframes(),
        }


def _wire_stt_client(stub: StubOpenAI) -> SttClient:
    return SttClient("http://stub", client=stub.client())


def _wire_tts_client(stub: StubOpenAI) -> TtsClient:
    return TtsClient("http://stub", client=stub.client())


# ── SttClient wire tests ──────────────────────────────────────────────────────

async def test_stt_transcribe_posts_to_audio_transcriptions() -> None:
    stub = StubOpenAI()
    stt = _wire_stt_client(stub)
    pcm = _make_pcm()
    result = await stt.transcribe(pcm, sample_rate=16000)
    assert result == "stub-transcription"
    assert stub.last_request().url.path == "/v1/audio/transcriptions"


async def test_stt_transcribe_sends_wav_wrapped_audio() -> None:
    """PCM bytes must be wrapped in a WAV container before posting."""
    stub = StubOpenAI()
    stt = _wire_stt_client(stub)
    pcm = _make_pcm(320)
    await stt.transcribe(pcm, sample_rate=16000, channels=1)

    req = stub.last_request()
    # The request is multipart — find the WAV file part in the body.
    content_type = req.headers["content-type"]
    assert "multipart/form-data" in content_type

    body = req.read()
    # The WAV magic bytes RIFF must appear somewhere in the multipart body.
    assert b"RIFF" in body
    assert b"WAVE" in body


async def test_stt_transcribe_wav_header_matches_pcm_params() -> None:
    """WAV header sample_rate and channels must reflect the values passed in."""
    stub = StubOpenAI()
    stt = _wire_stt_client(stub)
    n_frames = 480
    pcm = _make_pcm(n_frames)
    await stt.transcribe(pcm, sample_rate=16000, channels=1)

    body = stub.last_request().read()
    # Isolate WAV bytes from the multipart boundary — same extraction as the
    # SDK-level STT tests: split on the first CRLF to drop any trailing boundary.
    riff_offset = body.index(b"RIFF")
    wav_bytes = body[riff_offset:].split(b"\r\n", 1)[0]
    hdr = _parse_wav_header(wav_bytes)
    assert hdr["sample_rate"]  == 16000
    assert hdr["channels"]     == 1
    assert hdr["sample_width"] == 2        # 16-bit PCM
    assert hdr["n_frames"]     == n_frames


async def test_stt_transcribe_multipart_includes_response_format_json() -> None:
    """form-data must include response_format=json (server needs it)."""
    stub = StubOpenAI()
    stt = _wire_stt_client(stub)
    await stt.transcribe(_make_pcm(), sample_rate=16000)

    body = stub.last_request().read()
    assert b"response_format" in body
    assert b"json" in body


async def test_stt_transcribe_returns_text_field() -> None:
    stub = StubOpenAI()
    stub.set_transcribe_text("hello world")
    stt = _wire_stt_client(stub)
    result = await stt.transcribe(_make_pcm(), sample_rate=8000)
    assert result == "hello world"


async def test_stt_constructor_signature_preserved() -> None:
    """SttClient(base_url, timeout) must work without keyword-only args."""
    client = SttClient("http://localhost:8103", 45.0)
    assert client._stt is not None
    await client.close()


# ── TtsClient wire tests ──────────────────────────────────────────────────────

async def test_tts_synthesize_posts_to_audio_speech() -> None:
    stub = StubOpenAI()
    tts = _wire_tts_client(stub)
    await tts.synthesize("hello")
    assert stub.last_request().url.path == "/v1/audio/speech"


async def test_tts_synthesize_sends_json_body_with_input_and_response_format() -> None:
    """Wire body must be {input: text, response_format: wav}."""
    stub = StubOpenAI()
    tts = _wire_tts_client(stub)
    await tts.synthesize("speak this")

    body = json.loads(stub.bodies[-1].decode())
    assert body["input"]           == "speak this"
    assert body["response_format"] == "wav"


async def test_tts_synthesize_returns_raw_bytes() -> None:
    stub = StubOpenAI()
    stub.set_speech_bytes(b"RIFF\x00\x00\x00\x00WAVEdata")
    tts = _wire_tts_client(stub)
    result = await tts.synthesize("test")
    assert result == b"RIFF\x00\x00\x00\x00WAVEdata"


async def test_tts_constructor_signature_preserved() -> None:
    """TtsClient(base_url, timeout) must work without keyword-only args."""
    client = TtsClient("http://localhost:8105", 45.0)
    assert client._tts is not None
    await client.close()


# ── context manager protocol ──────────────────────────────────────────────────

async def test_stt_async_context_manager() -> None:
    stub = StubOpenAI()
    async with _wire_stt_client(stub) as stt:
        result = await stt.transcribe(_make_pcm(), sample_rate=16000)
    assert result == "stub-transcription"


async def test_tts_async_context_manager() -> None:
    stub = StubOpenAI()
    async with _wire_tts_client(stub) as tts:
        result = await tts.synthesize("hi")
    assert result == stub._speech_bytes
