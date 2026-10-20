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
import yaml
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
    state = {"status": 200, "authorization": "Bearer inference-token"}
    monkeypatch.setenv("NIM_TEST_KEY", "inference-token")

    def handle(request):
        assert request.headers.get("Authorization") == state["authorization"]
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
    assert set(config.entries) == {"llm", "vlm", "stt", "tts", "embedding", "ocr"}
    assert all(spec.deployment.ownership == "external" for spec in config.entries.values())
    assert config.required_credentials == ("NIM_ENDPOINT_API_KEY",)
    assert config.embedding("embedding").kind == "openai_compat"
    assert config.embedding("embedding").model_name == "nvidia/nemotron-3-embed-1b"
    assert config.embedding("embedding").default_extras == {"input_type": "passage"}
    assert config.stt("stt").kind == config.tts("tts").kind == "riva_grpc"
    assert config.ocr("ocr").kind == "nemotron_ocr"
    assert config.ocr("ocr").health_path == "/v1/health/ready"
    assert config.ocr("ocr").base_url == "https://ocr.example.com"
    assert config.ocr("ocr").deployment.service is None
    for path in ROOT.glob("agent-samples/*/worker/pyproject.toml"):
        project = tomllib.loads(path.read_text())["project"]
        if any(dep.startswith("xr-ai-models") for dep in project["dependencies"]):
            assert "xr-ai-models[riva]" in project["dependencies"]


@pytest.mark.parametrize("native_nim", [False, True])
async def test_tea_embedding_profiles_use_native_dimensions_and_retrieval_inputs(endpoint, monkeypatch, native_nim):
    sample = ROOT / "agent-samples/tea-making-sample/yaml"
    rag = yaml.safe_load((sample / "rag_service.yaml").read_text())
    assert rag["embedding_dim"] == 2048
    profile = ROOT / "deployment/nim/models.yaml" if native_nim else sample / rag["models_config"]
    calls, state = endpoint
    monkeypatch.setenv("NIM_ENDPOINT_API_KEY", "inference-token")
    state["authorization"] = "Bearer inference-token" if native_nim else None
    async with make_embedding(load_models_config(profile), rag["embedding_role"]) as client:
        await client.embed(["query: How long should tea steep?", "passage: Steep tea for three minutes."])
    if native_nim:
        assert calls == [
            {"model": "nvidia/nemotron-3-embed-1b", "input": ["How long should tea steep?"], "input_type": "query"},
            {"model": "nvidia/nemotron-3-embed-1b", "input": ["Steep tea for three minutes."], "input_type": "passage"},
        ]
    else:
        assert calls == [{
            "model": "embed",
            "input": ["query: How long should tea steep?", "passage: Steep tea for three minutes."],
        }]


@pytest.mark.parametrize("names, message", [
    ((), "at least one supported sample role"),
    (("summarizer", "planner"), "unsupported roles: ['planner', 'summarizer']"),
])
async def test_smoke_check_rejects_invalid_roles_before_connecting(names, message, monkeypatch, tmp_path):
    profile = tmp_path / "models.json"
    profile.write_text(json.dumps({"models": {name: {
        "category": "llm", "adapter": {"preset": "nemotron_omni"},
        "endpoint": {"base_url": "https://llm.example.com"},
    } for name in names}}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("connected before validation"))
    with pytest.raises(ValueError) as error:
        await SMOKE["check"](profile)
    assert message in str(error.value)


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


@pytest.mark.parametrize("readiness,status,with_text,expected_error", [
    ("health", 200, True, None),
    ("none", 200, True, None),
    ("health", 200, False, RuntimeError),
    ("health", 401, True, RuntimeError),
    ("none", 503, True, httpx.HTTPStatusError),
])
async def test_external_ocr_smoke_auth_inference_and_cleanup(
    monkeypatch, tmp_path, readiness, status, with_text, expected_error,
):
    import yaml

    real_client = httpx.AsyncClient
    opened, calls = [], []
    source = yaml.safe_load((ROOT / "deployment/nim/models.yaml").read_text())
    entry = source["models"]["ocr"]
    entry["endpoint"]["readiness"] = readiness
    profile = tmp_path / "models.yaml"
    profile.write_text(yaml.safe_dump({"models": {"ocr": entry}}))
    image = tmp_path / "text.png"
    image.write_bytes(SMOKE["_red_image"]())
    monkeypatch.setenv("NIM_ENDPOINT_API_KEY", "ocr-inference-token")

    def handle(request):
        calls.append(request)
        assert request.url.host == "ocr.example.com"
        assert request.headers["Authorization"] == "Bearer ocr-inference-token"
        if request.method == "GET":
            assert request.url.path == "/v1/health/ready"
            return httpx.Response(status)
        assert request.url.path == "/v1/ocr"
        assert json.loads(request.content)["merge_levels"] == ["word"]
        # Fixture is a transport test, not real OCR of the generated blank PNG.
        detections = [{
            "text_prediction": {"text": "温度 23.5", "confidence": 0.95},
            "bounding_box": {"points": [
                {"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2},
                {"x": 0.8, "y": 0.4}, {"x": 0.1, "y": 0.4},
            ]},
        }] if with_text else []
        return httpx.Response(status, json={"data": [{"index": 0, "text_detections": detections}]})

    def client(**kwargs):
        instance = real_client(transport=httpx.MockTransport(handle), **kwargs)
        opened.append(instance)
        return instance

    monkeypatch.setattr(httpx, "AsyncClient", client)
    if expected_error is None:
        await SMOKE["check"](profile, ocr_image=image)
    else:
        with pytest.raises(expected_error):
            await SMOKE["check"](profile, ocr_image=image)
    assert len(opened) == 1 and opened[0].is_closed
    assert [call.method for call in calls] == (
        ["POST"] if readiness == "none" else ["GET"] if status != 200 else ["GET", "POST"]
    )


async def test_native_riva_endpoint_uses_real_grpc_serialization(monkeypatch):
    import datetime
    import io
    import wave
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import aclosing

    import grpc
    from cryptography.hazmat.primitives import serialization
    from device_io_hub.transport.livekit import _tls
    from riva.client.proto import riva_asr_pb2 as asr
    from riva.client.proto import riva_asr_pb2_grpc as asr_grpc
    from riva.client.proto import riva_tts_pb2 as tts
    from riva.client.proto import riva_tts_pb2_grpc as tts_grpc
    from xr_ai_models import make_stt, make_tts

    calls = []
    pcm = b"\x01\x00" * 160
    now = datetime.datetime.now(datetime.timezone.utc)
    root, root_key = _tls._generate_root(now)
    certificate, key = _tls._generate_leaf(root, root_key, {"127.0.0.1"}, now)
    root_pem = root.public_bytes(serialization.Encoding.PEM)
    credentials = grpc.ssl_server_credentials([(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ), certificate.public_bytes(serialization.Encoding.PEM),
    )])
    ssl_credentials = grpc.ssl_channel_credentials

    def trusted_credentials(root_certificates=None, private_key=None, certificate_chain=None):
        return ssl_credentials(
            root_certificates=root_pem if root_certificates is None else root_certificates,
            private_key=private_key, certificate_chain=certificate_chain,
        )

    # Trust only this test's CA, retaining certificate/hostname verification.
    monkeypatch.setattr(grpc, "ssl_channel_credentials", trusted_credentials)

    class Speech(asr_grpc.RivaSpeechRecognitionServicer, tts_grpc.RivaSpeechSynthesisServicer):
        def Recognize(self, request, context):
            calls.append(("stt", request, dict(context.invocation_metadata())))
            return asr.RecognizeResponse(results=[asr.SpeechRecognitionResult(
                alternatives=[asr.SpeechRecognitionAlternative(transcript="ready")],
            )])

        def SynthesizeOnline(self, requests, context):
            if isinstance(requests, tts.SynthesizeSpeechRequest):
                requests = (requests,)
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
        port = server.add_secure_port("127.0.0.1:0", credentials)
        server.start()
        config = load_models_config_from_dict({name: {
            "category": name,
            "adapter": {"kind": "riva_grpc", "language": "en-US", "voice": "test-voice",
                        "sample_rate": 22050, "use_ssl": True},
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
    assert all(request.custom_dictionary == "" for kind, request, _ in calls if kind == "tts")


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
                assert await client.health()
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
        if request.url.path == "/v1/ocr":
            assert request.url.host == "ocr.example.com"
            assert payload["merge_levels"] == ["word"]
            assert len(payload["input"]) == 1
            assert payload["input"][0]["url"].startswith("data:image/png;base64,")
            assert "model" not in payload
            return httpx.Response(200, json={"data": [{"index": 0, "text_detections": []}]})
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
    assert len(calls) == 8
    llm = [payload for host, payload in calls if host == "llm.example.com"]
    assert len(llm) == 3
    chat, stream, tools = llm
    assert not chat.get("stream") and not chat.get("tools")
    assert stream["stream"] is True and not stream.get("tools")
    assert not tools.get("stream")
    assert tools["tools"][0]["function"]["name"] == "deployment_ready"
    vision = [payload for host, payload in calls if host == "vlm.example.com"]
    assert len(vision) == 2
    assert vision[0]["messages"][0]["content"][0]["type"] == "image_url"
