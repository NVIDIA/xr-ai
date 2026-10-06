# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private speaker-service HTTP identity and WebSocket stream contract."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from speaker_stt import __main__ as service
from speaker_stt import _inference
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import connect
from xr_ai_voicegate._speaker import _SpeakerConfig

_CFG = {
    "host": "127.0.0.1",
    "port": 8102,
    "asr_model": "test-asr",
    "diar_model": "test-diar",
}


class _Session:
    def __init__(self):
        self.count = 0

    def _feed(self, audio):
        self.count += 1
        return [{"kind": "transcript", "text": str(self.count)}]


def _models(factory=_Session):
    return SimpleNamespace(_session=lambda cfg, *, audio_origin_us: factory())


def _opening(*, audio_origin_us=1):
    return {"audio_origin_us": audio_origin_us, "config": asdict(_SpeakerConfig())}


def test_speaker_config_requires_an_http_origin():
    assert _SpeakerConfig().base_url == "http://127.0.0.1:8102"
    assert _SpeakerConfig._from_yaml({"enabled": True, "base_url": "http://localhost:8123"})
    assert _SpeakerConfig._from_yaml({"enabled": True, "base_url": "http://speaker.example:8123"})
    for base_url in (
        "https://127.0.0.1:8102",
        "http://127.0.0.1:8102/path",
        "http://127.0.0.1",
    ):
        with pytest.raises(ValueError, match="HTTP origin"):
            _SpeakerConfig._from_yaml({"enabled": True, "base_url": base_url})


@pytest.mark.parametrize("host", ["", 7])
def test_service_rejects_invalid_bind_hosts(host):
    with pytest.raises(ValueError, match="hostname or address"):
        service._base_url({"host": host, "port": 8102})


def test_service_allows_an_operator_selected_bind_host():
    assert service._base_url({"host": "0.0.0.0", "port": 8102}) == "http://127.0.0.1:8102"
    assert service._base_url({"host": "::", "port": 8102}) == "http://[::1]:8102"


def test_sessions_isolate_decoder_state_and_close_idempotently():
    server = service._Server(_models())
    first = server._open(_opening())
    second = server._open(_opening())
    assert server._audio(first, bytes(640))[0]["text"] == "1"
    assert server._audio(second, bytes(640))[0]["text"] == "1"
    assert server._audio(first, bytes(640))[0]["text"] == "2"
    for _ in range(2):
        server._close(first)
    assert list(server.sessions) == [second]


def test_session_expiry_and_capacity_are_bounded(monkeypatch):
    server = service._Server(_models(), max_sessions=1, idle_timeout_s=1)
    monkeypatch.setattr(service.time, "monotonic", lambda: 0)
    first = server._open(_opening())
    with pytest.raises(ValueError, match="capacity"):
        server._open(_opening())
    monkeypatch.setattr(service.time, "monotonic", lambda: 2)
    with pytest.raises(ValueError, match="unknown"):
        server._audio(first, bytes(640))
    assert not server.sessions
    assert server._open(_opening()) in server.sessions


@pytest.mark.parametrize("audio", ["pcm", bytes(32002)])
def test_invalid_audio_fails_before_inference(audio):
    server = service._Server(_models())
    session = server._open(_opening())
    with pytest.raises(ValueError, match="one second"):
        server._audio(session, audio)
    assert server.sessions[session][0].count == 0


@pytest.mark.parametrize("audio_origin_us", [None, -1, True, 1.5, "1"])
def test_open_requires_nonnegative_integer_audio_origin(audio_origin_us):
    server = service._Server(_models())
    with pytest.raises(ValueError, match="origin timestamp"):
        server._open(_opening(audio_origin_us=audio_origin_us))
    assert not server.sessions


def test_inference_failure_revokes_session():
    class Failing:
        def _feed(self, _audio):
            raise ValueError("model failure")

    server = service._Server(_models(Failing))
    session = server._open(_opening())
    with pytest.raises(ValueError, match="model failure"):
        server._audio(session, bytes(640))
    assert not server.sessions


def test_http_identity_models_and_websocket_session_lifecycle():
    server = service._Server(_models())
    app = service._build_app(server, _CFG)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health == service._identity(_CFG)
        assert client.get("/v1/models").json() == {
            "object": "list",
            "data": [
                {"id": "test-asr", "object": "model", "owned_by": "local"},
                {"id": "test-diar", "object": "model", "owned_by": "local"},
            ],
        }
        with client.websocket_connect(service._STREAM_PATH) as websocket:
            websocket.send_json(_opening(audio_origin_us=123))
            assert websocket.receive_json() == {
                "service": service._SERVICE,
                "protocol": service._PROTOCOL,
            }
            assert len(server.sessions) == 1
            websocket.send_bytes(bytes(640))
            assert websocket.receive_json() == {
                "events": [{"kind": "transcript", "text": "1"}],
            }
            websocket.send_bytes(bytes(640))
            assert websocket.receive_json()["events"][0]["text"] == "2"
    assert not server.sessions


def test_websocket_preserves_diagnostic_config_and_display_events():
    event = {"kind": "diagnostic", "speaker_id": 1, "status": "ignored", "pts_us": 123, "text": "hello"}
    observed = []

    def session(cfg, *, audio_origin_us):
        observed.append((cfg.diagnostics, audio_origin_us))
        return SimpleNamespace(_feed=lambda audio: [event])

    server = service._Server(SimpleNamespace(_session=session))
    with TestClient(service._build_app(server, _CFG)) as client:
        with client.websocket_connect(service._STREAM_PATH) as websocket:
            opening = _opening(audio_origin_us=123)
            opening["config"]["diagnostics"] = True
            websocket.send_json(opening)
            assert websocket.receive_json()["protocol"] == service._PROTOCOL
            websocket.send_bytes(bytes(640))
            assert websocket.receive_json() == {"events": [event]}
    assert observed == [(True, 123)]
    assert not server.sessions


def test_websocket_rejects_invalid_opening_and_text_audio():
    server = service._Server(_models())
    with TestClient(service._build_app(server, _CFG)) as client:
        with client.websocket_connect(service._STREAM_PATH) as websocket:
            websocket.send_json(_opening(audio_origin_us=-1))
            assert "origin timestamp" in websocket.receive_json()["error"]
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1008
        with client.websocket_connect(service._STREAM_PATH) as websocket:
            websocket.send_json(_opening())
            websocket.receive_json()
            websocket.send_text("not PCM")
            assert "binary PCM" in websocket.receive_json()["error"]
    assert not server.sessions


def test_websocket_inference_error_is_stable_and_revokes_session():
    class Failing:
        def _feed(self, _audio):
            raise ValueError("private model detail")

    server = service._Server(_models(Failing))
    with TestClient(service._build_app(server, _CFG)) as client:
        with client.websocket_connect(service._STREAM_PATH) as websocket:
            websocket.send_json(_opening())
            websocket.receive_json()
            websocket.send_bytes(bytes(640))
            assert websocket.receive_json() == {"error": "speaker inference failed"}
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1011
    assert not server.sessions


@pytest.mark.asyncio
async def test_cancelled_request_keeps_model_work_owned_until_completion():
    started = asyncio.Event()
    release = asyncio.Event()

    async def model_work():
        started.set()
        await release.wait()

    work = asyncio.create_task(model_work())
    request = asyncio.create_task(service._await_without_abandoning(work))
    await started.wait()
    request.cancel()
    await asyncio.sleep(0)
    assert not request.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert work.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "16"])
async def test_invalid_capacity_fails_before_model_loading_or_readiness(tmp_path, monkeypatch, capacity):
    def fail(_cfg):
        pytest.fail("invalid capacity reached model loading")

    monkeypatch.setattr(_inference, "_Models", fail)
    ready = tmp_path / "ready"
    with pytest.raises(ValueError, match="positive integer"):
        await service._serve({**_CFG, "max_sessions": capacity}, ready)
    assert not ready.exists()
    with pytest.raises(ValueError, match="positive integer"):
        service._Server(SimpleNamespace(), max_sessions=capacity)


@pytest.mark.asyncio
async def test_model_environment_precedence_and_readiness_cleanup(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setenv("HF_HOME", "operator-hf")
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.delenv("NEMO_CACHE_DIR", raising=False)
    monkeypatch.setattr(service, "_reuse", lambda *_args: asyncio.sleep(0, result=False))
    monkeypatch.setattr(
        _inference,
        "_Models",
        lambda cfg: loads.append(cfg) or _models(),
    )

    class Listener:
        def close(self):
            pass

    monkeypatch.setattr(service, "_listener", lambda _cfg: Listener())
    ready = tmp_path / "ready"
    observed = []

    class Config:
        def __init__(self, app, **_kwargs):
            self.app = app

    class UvicornServer:
        def __init__(self, config):
            self.config = config

        async def serve(self, *, sockets):
            observed.extend(sockets)
            async with self.config.app.router.lifespan_context(self.config.app):
                assert ready.exists()

    import uvicorn

    monkeypatch.setattr(uvicorn, "Config", Config)
    monkeypatch.setattr(uvicorn, "Server", UvicornServer)
    cache = tmp_path / "models"
    await service._serve({**_CFG, "model_cache": str(cache)}, ready)

    assert len(loads) == 1 and len(observed) == 1
    assert os.environ["HF_HOME"] == "operator-hf"
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["NEMO_CACHE_DIR"] == str((cache / "nemo").resolve())
    assert not ready.exists()


@pytest.mark.asyncio
async def test_real_loopback_server_exchanges_ordered_pcm(tmp_path, monkeypatch):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    monkeypatch.setattr(_inference, "_Models", lambda _cfg: _models())
    cfg = {
        **_CFG,
        "port": port,
        "model_cache": str(tmp_path / "models"),
    }
    ready = tmp_path / "ready"
    task = asyncio.create_task(service._serve(cfg, ready))
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
        status = await service._status(f"http://127.0.0.1:{port}")
        assert status == service._identity(cfg)
        monkeypatch.setenv(service._READY_PROCESS_MAY_EXIT_ENV, "1")
        reused_ready = tmp_path / "reused.ready"
        await service._serve(cfg, reused_ready)
        assert reused_ready.exists()
        with pytest.raises(RuntimeError, match="configuration changed"):
            await service._serve({**cfg, "precision": "float32"}, tmp_path / "changed.ready")
        uri = f"ws://127.0.0.1:{port}{service._STREAM_PATH}"
        async with connect(uri) as websocket:
            await websocket.send(json.dumps(_opening(audio_origin_us=123)))
            assert json.loads(await websocket.recv()) == {
                "service": service._SERVICE,
                "protocol": service._PROTOCOL,
            }
            await websocket.send(bytes(640))
            assert json.loads(await websocket.recv()) == {
                "events": [{"kind": "transcript", "text": "1"}],
            }
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not ready.exists()


def test_wrapper_execs_owned_listener_with_configured_port(tmp_path, monkeypatch):
    config = tmp_path / "speaker.yaml"
    config.write_text(yaml.safe_dump({**_CFG, "port": 8123, "model_cache": "models"}))
    monkeypatch.setattr(service, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["speaker_stt", "--config", str(config)])
    captured = {}

    class Executed(Exception):
        pass

    def execvpe(executable, argv, env):
        captured.update(executable=executable, argv=argv, env=env)
        raise Executed

    monkeypatch.setattr(service.os, "execvpe", execvpe)
    with pytest.raises(Executed):
        service._run()

    assert captured["executable"] == sys.executable
    assert captured["argv"][-2:] == ["--config", str(config)]
    assert captured["env"]["XR_AI_VLLM_MANAGED"] == "1"
    assert captured["env"]["XR_AI_VLLM_PORT"] == "8123"
    assert service._READY_PROCESS_MAY_EXIT_ENV not in captured["env"]


def test_status_refuses_foreign_or_old_protocol_endpoint(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(service.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    for identity in (
        {"service": "foreign", "protocol": service._PROTOCOL},
        {"service": service._SERVICE, "protocol": service._PROTOCOL - 1},
    ):
        monkeypatch.setattr(service.json, "load", lambda _response, value=identity: value)
        with pytest.raises(RuntimeError, match="incompatible"):
            service._blocking_status("http://127.0.0.1:8102")


@pytest.mark.asyncio
async def test_reuse_watcher_requires_three_consecutive_status_failures(monkeypatch):
    expected = {"service": service._SERVICE, "protocol": service._PROTOCOL, "fingerprint": "same"}
    responses = iter([None, None, expected, None, None, None])
    calls = []

    async def status(base_url):
        calls.append(base_url)
        return next(responses)

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(service, "_status", status)
    monkeypatch.setattr(service.asyncio, "sleep", no_sleep)
    with pytest.raises(RuntimeError, match="no longer available"):
        await service._watch_reused("http://127.0.0.1:8102", "same")
    assert len(calls) == 6
