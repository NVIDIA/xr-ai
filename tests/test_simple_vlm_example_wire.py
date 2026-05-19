# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-format golden tests for simple-vlm-example's STT/VLM/TTS client construction.

Verifies that the SDK-backed clients send the same HTTP body shapes that the
hand-rolled services.py sent before the migration.  No GPU or real server
required — all assertions run against StubOpenAI (httpx.MockTransport).
"""
from __future__ import annotations

import base64
import io
import json
import wave

from _stub_openai import StubOpenAI
from xr_ai_models import (
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
    load_models_config,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _make_wav(sample_rate: int = 16000, n_samples: int = 160) -> bytes:
    """Build a minimal valid WAV file with silence for test input."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def _make_data_url(pixel_bytes: bytes = b"\xff\xd8\xff\xe0stub") -> str:
    """Build a minimal data: URL to simulate encode_image() output."""
    b64 = base64.b64encode(pixel_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"


# ── STT golden ─────────────────────────────────────────────────────────────


async def test_stt_request_shape() -> None:
    """STT sends a multipart POST to /v1/audio/transcriptions with file= shape."""
    stub = StubOpenAI()
    stub.set_transcribe_text("hello world")

    stt = OpenAICompatSTT("http://stub", client=stub.client())
    wav = _make_wav()
    text = await stt.transcribe(wav)

    assert text == "hello world"

    req = stub.last_request()
    assert req.url.path == "/v1/audio/transcriptions"
    # multipart body must contain file field and response_format
    body = req.content.decode(errors="replace")
    assert "audio.wav" in body
    assert "audio/wav" in body
    assert "response_format" in body
    assert "json" in body


async def test_stt_transcribe_passes_wav_bytes_unchanged() -> None:
    """When audio is already WAV bytes (no sample_rate kwarg), it's sent as-is."""
    stub = StubOpenAI()
    stt = OpenAICompatSTT("http://stub", client=stub.client())

    wav = _make_wav(sample_rate=16000, n_samples=320)
    await stt.transcribe(wav)

    # The body should contain RIFF header because we passed pre-formed WAV.
    req = stub.last_request()
    assert b"RIFF" in req.content


# ── VLM golden ─────────────────────────────────────────────────────────────


async def test_vlm_stream_request_shape() -> None:
    """VLM sends streaming POST to /v1/chat/completions with image content part."""
    stub = StubOpenAI()
    stub.set_stream_tokens(["Hello", " world"])

    # cosmos_vlm preset sets enable_thinking=False via default_extras.
    vlm = OpenAICompatVLM(
        "http://stub",
        "vlm",
        default_extras={"chat_template_kwargs": {"enable_thinking": False}},
        client=stub.client(),
    )
    image_url = _make_data_url()
    tokens = []
    async for tok in vlm.stream(image_url, "What do you see?",
                                system_prompt="You are helpful."):
        tokens.append(tok)

    assert tokens == ["Hello", " world"]

    req = stub.last_request()
    assert req.url.path == "/v1/chat/completions"
    body = json.loads(req.content)
    assert body["stream"] is True

    # System prompt must appear as the first message.
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are helpful."

    # User message must contain an image_url part and a text part.
    user_msg = body["messages"][1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    image_parts = [p for p in content if p.get("type") == "image_url"]
    text_parts  = [p for p in content if p.get("type") == "text"]
    assert len(image_parts) == 1
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "What do you see?"

    # Cosmos preset injects enable_thinking=False via chat_template_kwargs.
    assert body.get("chat_template_kwargs", {}).get("enable_thinking") is False


async def test_vlm_stream_no_system_prompt() -> None:
    """VLM omits the system message when system_prompt is empty."""
    stub = StubOpenAI()
    stub.set_stream_tokens(["ok"])

    vlm = OpenAICompatVLM("http://stub", "vlm", client=stub.client())
    async for _ in vlm.stream(_make_data_url(), "ping"):
        pass

    body = json.loads(stub.last_request().content)
    assert all(m["role"] != "system" for m in body["messages"])


# ── TTS golden ─────────────────────────────────────────────────────────────


async def test_tts_request_shape() -> None:
    """TTS sends JSON POST to /v1/audio/speech with input and response_format=wav."""
    stub = StubOpenAI()
    stub.set_speech_bytes(b"RIFF\x00\x00\x00\x00WAVEstub")

    tts = OpenAICompatTTS("http://stub", client=stub.client())
    audio = await tts.synthesize("Hello there.")

    assert audio == b"RIFF\x00\x00\x00\x00WAVEstub"

    req = stub.last_request()
    assert req.url.path == "/v1/audio/speech"
    body = json.loads(req.content)
    assert body["input"] == "Hello there."
    assert body["response_format"] == "wav"


# ── models.yaml factory round-trip ─────────────────────────────────────────
#
# These tests verify that load_models_config + make_* resolve presets correctly
# and produce clients with the right base URLs and model parameters.  Wire
# assertions use the public OpenAICompat* constructors with an injected stub
# client rather than reaching into private attributes of factory-built objects.


def test_make_stt_from_config_resolves_preset(tmp_path) -> None:
    """parakeet_stt preset resolves to an STT spec with the supplied base_url."""
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "stt:\n"
        "  kind: preset:parakeet_stt\n"
        "  base_url: http://localhost:8103\n"
    )
    cfg = load_models_config(models_yaml)
    spec = cfg.stt("stt")
    assert spec.base_url == "http://localhost:8103"


def test_make_vlm_from_config_cosmos_preset_has_enable_thinking_false(tmp_path) -> None:
    """cosmos_vlm preset resolves with enable_thinking=False in default_extras."""
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "vlm:\n"
        "  kind: preset:cosmos_vlm\n"
        "  base_url: http://localhost:8100\n"
    )
    cfg  = load_models_config(models_yaml)
    spec = cfg.vlm("vlm")
    assert spec.default_extras.get("chat_template_kwargs", {}).get("enable_thinking") is False


def test_make_tts_from_config_resolves_preset(tmp_path) -> None:
    """piper_tts preset resolves to a TTS spec with the supplied base_url."""
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "tts:\n"
        "  kind: preset:piper_tts\n"
        "  base_url: http://localhost:8105\n"
    )
    cfg  = load_models_config(models_yaml)
    spec = cfg.tts("tts")
    assert spec.base_url == "http://localhost:8105"
