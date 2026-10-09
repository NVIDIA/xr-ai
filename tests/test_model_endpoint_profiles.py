# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native endpoint profiles use the same factories and preserve local payloads."""
from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from runpy import run_path

import httpx
import pytest
from xr_ai_models import load_models_config, load_models_config_from_dict, make_embedding

ROOT = Path(__file__).resolve().parents[1]
SMOKE = run_path(str(ROOT / "deployment/nim/smoke_test.py"))


def _profile(extras):
    return load_models_config_from_dict({"embedding": {
        "category": "embedding",
        "adapter": {"kind": "openai_compat", "model_name": "native-model", "default_extras": extras},
        "endpoint": {"base_url": "https://embedding.example.com", "api_key_env": "NIM_TEST_KEY",
                     "health_path": "/v1/health/ready"},
    }})


@pytest.fixture
def endpoint(monkeypatch):
    real_client = httpx.AsyncClient
    calls = []
    state = {"status": 200}
    monkeypatch.setenv("NIM_TEST_KEY", "inference-token")

    def handle(request):
        assert request.headers["Authorization"] == "Bearer inference-token"
        if request.url.path == "/v1/health/ready":
            return httpx.Response(state["status"], json={"status": "ready"})
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content)
        calls.append(payload)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"detail": "unavailable"})
        return httpx.Response(200, json={"data": [
            {"index": index, "embedding": [float(len(text))]}
            for index, text in reversed(list(enumerate(payload["input"])))
        ]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    return calls, state


async def test_asymmetric_batches_send_native_payloads_and_restore_order(endpoint):
    calls, _ = endpoint
    client = make_embedding(_profile({"input_type": "passage", "truncate": "NONE"}), "embedding")
    try:
        assert await client.health()
        assert await client.embed(["passage: abcd", "query: hi", "plain", "query: z"]) == [
            [4.0], [2.0], [5.0], [1.0],
        ]
        assert calls == [
            {"model": "native-model", "input": ["hi", "z"], "input_type": "query", "truncate": "NONE"},
            {"model": "native-model", "input": ["abcd", "plain"], "input_type": "passage", "truncate": "NONE"},
        ]
        assert await client.embed([]) == []
        assert len(calls) == 2
    finally:
        await client.close()
    assert client._client.is_closed


async def test_local_embedding_text_and_other_defaults_are_preserved(endpoint):
    calls, _ = endpoint
    for extras in ({}, {"dimensions": 768}):
        async with make_embedding(_profile(extras), "embedding") as client:
            assert await client.embed(["query: alpha", "passage: beta"]) == [[12.0], [13.0]]
        assert calls[-1] == {"model": "native-model", "input": ["query: alpha", "passage: beta"], **extras}


@pytest.mark.parametrize("extras", [{"input_type": "document"}, {"model": "other"}, {"input": ["other"]}])
def test_invalid_embedding_defaults_fail_before_opening_client(extras, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("client was opened"))
    with pytest.raises(ValueError):
        make_embedding(_profile(extras), "embedding")


async def test_native_errors_propagate_and_client_closes(endpoint):
    _, state = endpoint
    state["status"] = 503
    client = make_embedding(_profile({"input_type": "passage"}), "embedding")
    with pytest.raises(httpx.HTTPStatusError):
        async with client:
            assert not await client.health()
            await client.embed(["query: question"])
    assert client._client.is_closed


async def test_embedding_cancellation_does_not_submit_second_group(monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    calls = []
    real_client = httpx.AsyncClient

    async def handle(request):
        calls.append(json.loads(request.content))
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    client = make_embedding(_profile({"input_type": "passage"}), "embedding")
    async with client:
        task = asyncio.create_task(client.embed(["query: q", "passage: p"]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert len(calls) == 1
    assert client._client.is_closed


def test_endpoint_profile_and_sample_environments_use_existing_contract():
    config = load_models_config(ROOT / "deployment/nim/models.yaml")
    assert set(config.entries) == {"llm", "agent_llm", "vlm", "stt", "tts", "embedding"}
    assert all(spec.deployment.ownership == "external" for spec in config.entries.values())
    assert config.required_credentials == ("NIM_ENDPOINT_API_KEY",)
    assert config.embedding("embedding").kind == "openai_compat"
    assert config.embedding("embedding").default_extras == {"input_type": "passage"}
    assert config.stt("stt").kind == config.tts("tts").kind == "riva_grpc"
    for path in ROOT.glob("agent-samples/*/worker/pyproject.toml"):
        project = tomllib.loads(path.read_text())["project"]
        if any(dep.startswith("xr-ai-models") for dep in project["dependencies"]):
            assert "xr-ai-models[riva]" in project["dependencies"]


async def test_smoke_check_rejects_missing_credentials_before_connecting(monkeypatch):
    monkeypatch.delenv("NIM_ENDPOINT_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("connected before validation"))
    with pytest.raises(ValueError, match="missing endpoint credential: NIM_ENDPOINT_API_KEY"):
        await SMOKE["check"](ROOT / "deployment/nim/models.yaml")


async def test_smoke_check_accepts_native_embedding_dimensions(endpoint, tmp_path, capsys):
    profile = tmp_path / "models.json"
    profile.write_text(json.dumps({"embedding": {
        "category": "embedding", "adapter": {"model_name": "native-model", "default_extras": {"input_type": "passage"}},
        "endpoint": {"base_url": "https://embedding.example.com", "api_key_env": "NIM_TEST_KEY",
                     "health_path": "/v1/health/ready"},
    }}))
    await SMOKE["check"](profile)
    assert "1-dimensional vectors" in capsys.readouterr().out


async def test_native_riva_endpoint_uses_real_grpc_serialization(monkeypatch):
    import io
    import wave
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import aclosing

    import grpc
    from riva.client.proto import riva_asr_pb2 as asr
    from riva.client.proto import riva_asr_pb2_grpc as asr_grpc
    from riva.client.proto import riva_tts_pb2 as tts
    from riva.client.proto import riva_tts_pb2_grpc as tts_grpc
    from xr_ai_models import make_stt, make_tts

    calls = []
    pcm = b"\x01\x00" * 160

    class Speech(asr_grpc.RivaSpeechRecognitionServicer, tts_grpc.RivaSpeechSynthesisServicer):
        def Recognize(self, request, context):
            calls.append(("stt", request, dict(context.invocation_metadata())))
            return asr.RecognizeResponse(results=[asr.SpeechRecognitionResult(
                alternatives=[asr.SpeechRecognitionAlternative(transcript="ready")],
            )])

        def SynthesizeOnline(self, requests, context):
            for request in requests:
                calls.append(("tts", request, dict(context.invocation_metadata())))
                yield tts.SynthesizeSpeechResponse(audio=pcm[:1])
                yield tts.SynthesizeSpeechResponse(audio=pcm[1:])

    monkeypatch.setenv("NIM_TEST_KEY", "inference-token")
    with ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool)
        peer = Speech()
        asr_grpc.add_RivaSpeechRecognitionServicer_to_server(peer, server)
        tts_grpc.add_RivaSpeechSynthesisServicer_to_server(peer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        config = load_models_config_from_dict({name: {
            "category": name,
            "adapter": {"kind": "riva_grpc", "language": "en-US", "voice": "test-voice", "sample_rate": 22050},
            "endpoint": {"base_url": f"127.0.0.1:{port}", "api_key_env": "NIM_TEST_KEY"},
        } for name in ("stt", "tts")})
        try:
            async with make_stt(config, "stt") as recognition, make_tts(config, "tts") as synthesis:
                assert await recognition.health()
                assert await synthesis.health()
                assert await recognition.transcribe(pcm, sample_rate=16000) == "ready"
                audio = await synthesis.synthesize("ready")
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    assert wav.getframerate() == 22050
                    assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
                    assert wav.readframes(wav.getnframes()) == pcm
                async with aclosing(synthesis.stream("ready")) as stream:
                    chunks = [chunk async for chunk in stream]
                assert b"".join(chunk.data for chunk in chunks) == pcm
                assert all(chunk.sample_rate == 22050 and chunk.channels == 1 for chunk in chunks)
        finally:
            server.stop(0).wait()
    assert [kind for kind, _, _ in calls] == ["stt", "tts", "tts"]
    assert all(metadata["authorization"] == "Bearer inference-token" for _, _, metadata in calls)
    assert calls[0][1].config.sample_rate_hertz == 16000
    assert calls[1][1].voice_name == "test-voice"


@pytest.mark.parametrize("action", ["cancel", "timeout"])
async def test_native_recognition_releases_rpc_before_returning(action):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from riva.client.proto import riva_asr_pb2 as asr
    from riva.client.proto import riva_asr_pb2_grpc as asr_grpc
    from xr_ai_models import make_stt

    received = threading.Event()
    departed = threading.Event()

    class Recognition(asr_grpc.RivaSpeechRecognitionServicer):
        def Recognize(self, request, context):
            context.add_callback(departed.set)
            received.set()
            departed.wait()
            return asr.RecognizeResponse()

    with ThreadPoolExecutor(max_workers=1) as pool:
        server = grpc.server(pool)
        asr_grpc.add_RivaSpeechRecognitionServicer_to_server(Recognition(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        config = load_models_config_from_dict({"stt": {
            "category": "stt", "adapter": {"kind": "riva_grpc"},
            "endpoint": {"base_url": f"127.0.0.1:{port}"},
        }})
        try:
            async with make_stt(config, "stt") as client:
                task = asyncio.create_task(client.transcribe(
                    b"\x01\x00" * 160, sample_rate=16000,
                    timeout=0.1 if action == "timeout" else 30,
                ))
                assert await asyncio.to_thread(received.wait, 5)
                if action == "cancel":
                    task.cancel()
                with pytest.raises(asyncio.CancelledError if action == "cancel" else TimeoutError):
                    await task
                assert await asyncio.to_thread(departed.wait, 5)
        finally:
            server.stop(0).wait()
            departed.set()


async def test_smoke_uses_native_http_profile_for_chat_vision_and_embeddings(monkeypatch, tmp_path):
    import yaml

    real_client = httpx.AsyncClient
    calls = []
    source = yaml.safe_load((ROOT / "deployment/nim/models.yaml").read_text())
    source["models"] = {name: spec for name, spec in source["models"].items() if name not in ("stt", "tts")}
    profile = tmp_path / "models.yaml"
    profile.write_text(yaml.safe_dump(source))
    monkeypatch.setenv("NIM_ENDPOINT_API_KEY", "native-inference-token")

    def handle(request):
        assert request.headers["Authorization"] == "Bearer native-inference-token"
        if request.url.path == "/v1/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        payload = json.loads(request.content)
        calls.append((request.url.host, payload))
        assert payload["model"].startswith("nvidia/")
        if request.url.path == "/v1/embeddings":
            assert payload["input_type"] in ("query", "passage")
            assert not any(text.startswith(("query: ", "passage: ")) for text in payload["input"])
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [1.0, 2.0]} for index, _ in enumerate(payload["input"])
            ]})
        assert request.url.path == "/v1/chat/completions"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if payload.get("stream"):
            event = {"choices": [{"delta": {"content": "ready"}}]}
            return httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        message = {"content": "ready"}
        if payload.get("tools"):
            message = {"content": None, "tool_calls": [{
                "id": "ready-call", "type": "function", "function": {"name": "deployment_ready", "arguments": "{}"},
            }]}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    await SMOKE["check"](profile)
    assert len(calls) == 10
    vision = [payload for host, payload in calls if host == "vlm.example.com"]
    assert len(vision) == 2
    assert vision[0]["messages"][0]["content"][0]["type"] == "image_url"
