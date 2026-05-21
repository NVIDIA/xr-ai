# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-format and behavior coverage for the OpenAICompat* clients.

Uses :class:`_stub_openai.StubOpenAI` (an ``httpx.MockTransport``) so no
real HTTP listener is needed.
"""
from __future__ import annotations

import base64
import io
import wave
import pytest

from _stub_openai import StubOpenAI

from xr_ai_models import (
    Capabilities,
    ChatMessage,
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
    ToolDef,
)


# ── LLM: chat ─────────────────────────────────────────────────────────────


async def test_llm_chat_builds_minimal_payload() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(content="hi there")
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        resp = await llm.chat([ChatMessage(role="user", content="hello")])

    assert resp.content == "hi there"
    assert resp.reasoning is None
    assert resp.tool_calls is None
    assert resp.finish_reason == "stop"

    body = stub.last_json()
    assert body["model"]    == "llm"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "stream" not in body


async def test_llm_chat_passes_through_max_tokens_temperature() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        await llm.chat(
            [ChatMessage(role="user", content="hello")],
            max_tokens=40, temperature=0.7,
        )
    body = stub.last_json()
    assert body["max_tokens"]  == 40
    assert body["temperature"] == 0.7


async def test_llm_chat_folds_thinking_kwargs_into_chat_template_kwargs() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        await llm.chat(
            [ChatMessage(role="user", content="x")],
            enable_thinking=True, thinking_budget=1024,
        )
    body = stub.last_json()
    assert body["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking_budget": 1024,
    }


async def test_llm_chat_default_extras_merged_with_per_call() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm",
        default_extras={"chat_template_kwargs": {"enable_thinking": False}},
        client=stub.client(),
    ) as llm:
        await llm.chat(
            [ChatMessage(role="user", content="x")],
            thinking_budget=256,
        )
    body = stub.last_json()
    assert body["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking_budget": 256,
    }


async def test_llm_chat_reasoning_field_aliasing() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(
        content="answer",
        reasoning="thinking…",
        reasoning_field="reasoning_content",
    )
    async with OpenAICompatLLM(
        "http://stub", "llm",
        reasoning_field="reasoning_content",
        client=stub.client(),
    ) as llm:
        resp = await llm.chat([ChatMessage(role="user", content="x")])
    assert resp.reasoning == "thinking…"


async def test_llm_chat_reasoning_default_checks_both_field_names() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(
        content="answer", reasoning="r1", reasoning_field="reasoning_content",
    )
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        r1 = await llm.chat([ChatMessage(role="user", content="x")])
    assert r1.reasoning == "r1"

    stub.set_chat_message(
        content="a2", reasoning="r2", reasoning_field="reasoning",
    )
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        r2 = await llm.chat([ChatMessage(role="user", content="x")])
    assert r2.reasoning == "r2"


async def test_llm_chat_tool_definitions_serialized_to_openai_shape() -> None:
    stub = StubOpenAI()
    tool = ToolDef(
        name="get_weather",
        description="Get current weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        await llm.chat([ChatMessage(role="user", content="x")], tools=[tool])
    body = stub.last_json()
    assert body["tools"] == [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


async def test_llm_chat_tool_calls_parsed_from_response() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(
        content="",
        tool_calls=[{
            "id":   "call_abc",
            "type": "function",
            "function": {
                "name":      "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        }],
        finish_reason="tool_calls",
    )
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        resp = await llm.chat([ChatMessage(role="user", content="weather?")])
    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id        == "call_abc"
    assert tc.name      == "get_weather"
    assert tc.arguments == '{"city":"Paris"}'
    assert resp.finish_reason == "tool_calls"


async def test_llm_chat_assistant_tool_call_only_turn_sends_null_content() -> None:
    """Assistant turn that carries only tool_calls (no text) must serialize
    ``content: null`` on the wire, not ``content: ""`` — the OpenAI spec allows
    null and some vLLM versions reject the empty-string form. Mirrors the
    pre-SDK ``content or None`` wire shape.
    """
    from xr_ai_models import ToolCall  # local import: only used in this case

    stub = StubOpenAI()
    stub.set_chat_message(content="follow-up")
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        await llm.chat([
            ChatMessage(role="user", content="weather in Paris?"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="call_abc",
                    name="get_weather",
                    arguments='{"city":"Paris"}',
                )],
            ),
            ChatMessage(role="tool", content="22C, clear", tool_call_id="call_abc"),
        ])
    body = stub.last_json()
    assistant_msg = body["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    # And a plain text turn still gets its content through unchanged.
    user_msg = body["messages"][0]
    assert user_msg["content"] == "weather in Paris?"


async def test_llm_chat_sends_bearer_when_api_key_env_set(monkeypatch) -> None:
    monkeypatch.setenv("STUB_API_KEY", "sk-test-123")
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm",
        api_key_env="STUB_API_KEY",
        client=stub.client(),
    ) as llm:
        await llm.chat([ChatMessage(role="user", content="x")])
    assert stub.last_request().headers["Authorization"] == "Bearer sk-test-123"


async def test_llm_chat_no_auth_header_without_api_key_env() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        await llm.chat([ChatMessage(role="user", content="x")])
    assert "Authorization" not in stub.last_request().headers


async def test_llm_chat_raises_on_http_error() -> None:
    import httpx
    stub = StubOpenAI()
    stub.set_chat_status(500)
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        with pytest.raises(httpx.HTTPStatusError):
            await llm.chat([ChatMessage(role="user", content="x")])


# ── LLM: stream ───────────────────────────────────────────────────────────


async def test_llm_stream_yields_content_tokens_and_stops_at_done() -> None:
    stub = StubOpenAI()
    stub.set_stream_tokens(["Hel", "lo", " world"])
    chunks: list[str] = []
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        async for tok in llm.stream([ChatMessage(role="user", content="x")]):
            chunks.append(tok)
    assert chunks == ["Hel", "lo", " world"]
    body = stub.last_json()
    assert body["stream"] is True


# ── LLM: health ───────────────────────────────────────────────────────────


async def test_llm_health_true_on_200() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        assert (await llm.health()) is True


async def test_llm_health_false_on_503() -> None:
    stub = StubOpenAI()
    stub.set_health_status(503)
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
    ) as llm:
        assert (await llm.health()) is False


# ── VLM ───────────────────────────────────────────────────────────────────


_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


async def test_vlm_ask_image_with_bytes_builds_data_url() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(content="it's a cat")
    async with OpenAICompatVLM(
        "http://stub", "vlm", client=stub.client(),
    ) as vlm:
        resp = await vlm.ask_image(_PNG_HEADER, "what is this?")
    assert resp.content == "it's a cat"

    body = stub.last_json()
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG_HEADER
    assert parts[1] == {"type": "text", "text": "what is this?"}


async def test_vlm_ask_image_with_path_reads_file(tmp_path) -> None:
    p = tmp_path / "img.png"
    p.write_bytes(_PNG_HEADER)
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm", client=stub.client(),
    ) as vlm:
        await vlm.ask_image(p, "?")
    body = stub.last_json()
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


async def test_vlm_ask_image_with_string_passes_through_url() -> None:
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm", client=stub.client(),
    ) as vlm:
        await vlm.ask_image("https://example.com/img.png", "?")
    body = stub.last_json()
    assert body["messages"][0]["content"][0]["image_url"]["url"] == \
        "https://example.com/img.png"


async def test_vlm_ask_image_includes_system_prompt_when_set() -> None:
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm", client=stub.client(),
    ) as vlm:
        await vlm.ask_image(_PNG_HEADER, "?", system_prompt="be terse")
    body = stub.last_json()
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["messages"][1]["role"] == "user"


async def test_vlm_default_extras_propagate() -> None:
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm",
        default_extras={"chat_template_kwargs": {"enable_thinking": False}},
        client=stub.client(),
    ) as vlm:
        await vlm.ask_image(_PNG_HEADER, "?")
    body = stub.last_json()
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


# ── VLM: video ────────────────────────────────────────────────────────────


# ISO BMFF header: 4 bytes box size, "ftyp" magic, "isom" major brand,
# minor version, then "isom"/"mp42" compat brands. Smallest plausible mp4 prefix.
_MP4_HEADER = (
    b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00"
    b"isomiso2avc1mp41" + b"\x00" * 8
)
_WEBM_HEADER = b"\x1a\x45\xdf\xa3" + b"\x00" * 24


async def test_vlm_ask_video_with_bytes_sniffs_mp4_and_builds_data_url() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(content="a person walks left")
    async with OpenAICompatVLM(
        "http://stub", "vlm",
        capabilities=Capabilities(vision=True, video=True),
        client=stub.client(),
    ) as vlm:
        resp = await vlm.ask_video(_MP4_HEADER, "what happens?")
    assert resp.content == "a person walks left"

    body  = stub.last_json()
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "video_url"
    url = parts[0]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _MP4_HEADER
    assert parts[1] == {"type": "text", "text": "what happens?"}


async def test_vlm_ask_video_with_path_uses_suffix_mime(tmp_path) -> None:
    p = tmp_path / "clip.webm"
    p.write_bytes(_WEBM_HEADER)
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm",
        capabilities=Capabilities(vision=True, video=True),
        client=stub.client(),
    ) as vlm:
        await vlm.ask_video(p, "?")
    body = stub.last_json()
    url = body["messages"][0]["content"][0]["video_url"]["url"]
    assert url.startswith("data:video/webm;base64,")


async def test_vlm_ask_video_with_string_passes_through_url() -> None:
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm",
        capabilities=Capabilities(vision=True, video=True),
        client=stub.client(),
    ) as vlm:
        await vlm.ask_video("https://example.com/clip.mp4", "?")
    body = stub.last_json()
    assert body["messages"][0]["content"][0]["video_url"]["url"] == \
        "https://example.com/clip.mp4"


async def test_vlm_ask_video_raises_when_capability_off() -> None:
    stub = StubOpenAI()
    async with OpenAICompatVLM(
        "http://stub", "vlm",
        capabilities=Capabilities(vision=True),
        client=stub.client(),
    ) as vlm:
        with pytest.raises(ValueError, match="video"):
            await vlm.ask_video(_MP4_HEADER, "?")


# ── STT ───────────────────────────────────────────────────────────────────


async def test_stt_transcribe_wav_bytes_passthrough() -> None:
    stub = StubOpenAI()
    stub.set_transcribe_text("hello world")
    wav_bytes = b"RIFF\x00\x00\x00\x00WAVEdata"
    async with OpenAICompatSTT(
        "http://stub", client=stub.client(),
    ) as stt:
        text = await stt.transcribe(wav_bytes)
    assert text == "hello world"
    req = stub.last_request()
    assert req.url.path == "/v1/audio/transcriptions"
    assert wav_bytes in req.content


async def test_stt_transcribe_pcm_converts_to_wav() -> None:
    stub = StubOpenAI()
    pcm = b"\x00\x01" * 480
    async with OpenAICompatSTT(
        "http://stub", client=stub.client(),
    ) as stt:
        await stt.transcribe(pcm, sample_rate=16000, channels=1)
    sent = stub.last_request().content
    start = sent.find(b"RIFF")
    assert start >= 0
    extracted = sent[start:].split(b"\r\n", 1)[0]
    with wave.open(io.BytesIO(extracted), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1


# ── TTS ───────────────────────────────────────────────────────────────────


async def test_tts_synthesize_returns_audio_bytes() -> None:
    stub = StubOpenAI()
    stub.set_speech_bytes(b"\x01\x02\x03")
    async with OpenAICompatTTS(
        "http://stub", client=stub.client(),
    ) as tts:
        data = await tts.synthesize("hello")
    assert data == b"\x01\x02\x03"
    body = stub.last_json()
    assert body == {"input": "hello", "response_format": "wav"}


async def test_tts_response_format_override() -> None:
    stub = StubOpenAI()
    async with OpenAICompatTTS(
        "http://stub", client=stub.client(),
    ) as tts:
        await tts.synthesize("hi", response_format="pcm")
    body = stub.last_json()
    assert body["response_format"] == "pcm"


# ── lifecycle ─────────────────────────────────────────────────────────────


async def test_external_client_not_closed_by_close() -> None:
    stub = StubOpenAI()
    client = stub.client()
    llm = OpenAICompatLLM("http://stub", "llm", client=client)
    await llm.close()
    assert client.is_closed is False
    await client.aclose()


async def test_owned_client_closed_by_close() -> None:
    llm = OpenAICompatLLM("http://stub", "llm")
    inner = llm._client
    await llm.close()
    assert inner.is_closed is True
