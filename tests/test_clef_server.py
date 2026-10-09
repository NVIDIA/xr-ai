# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and config contract tests for the local Clef model server."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xr_ai_vllm import _docker, stop_persistent_servers

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "services" / "clef-server"))

import clef_server.__main__ as clef_main  # noqa: E402
from clef_server._config import ServerConfig, identity, load_config  # noqa: E402
from clef_server._service import (  # noqa: E402
    _await_without_abandoning,
    _normalize_response,
    create_app,
)

_SAMPLE_SPEC = importlib.util.spec_from_file_location(
    "clef_flash_sample",
    _REPO_ROOT / "model-server-samples" / "clef-flash" / "main.py",
)
assert _SAMPLE_SPEC and _SAMPLE_SPEC.loader
clef_sample = importlib.util.module_from_spec(_SAMPLE_SPEC)
_SAMPLE_SPEC.loader.exec_module(clef_sample)

REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"


class _Backend:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.revision = REVISION
        self.model = object()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.calls: list[dict] = []
        self.closed = False

    def validate_length(self, body: dict) -> None:
        if body["state"] == "too long":
            raise ValueError("request exceeds the configured token limit")

    def answer(self, body: dict) -> dict:
        self.calls.append(body)
        answers = {}
        for key, question in body["questions"].items():
            labels = list(question["criteria"])
            probabilities = {
                label: 0.9 if index == 0 else 0.1 / (len(labels) - 1)
                for index, label in enumerate(labels)
            }
            answers[key] = {
                "type": "choice",
                "choice": labels[0],
                "confidence": probabilities[labels[0]],
                "probabilities": probabilities,
            }
        return {
            "model": body["model"],
            "answers": answers,
            "usage": {"input_tokens": 3, "output_tokens": 0},
        }

    async def close(self) -> None:
        self.closed = True
        await __import__("asyncio").to_thread(self.executor.shutdown, wait=True)


def _config(tmp_path: Path, *, max_body_bytes: int = 1024) -> ServerConfig:
    return ServerConfig(
        model_name="Cloudflare/clef-flash",
        model_revision=REVISION,
        model_path=None,
        model_cache=tmp_path / "cache",
        host="127.0.0.1",
        port=8120,
        device="cuda:0",
        dtype="bfloat16",
        max_length=4096,
        max_body_bytes=max_body_bytes,
    )


def _request(**overrides) -> dict:
    body = {
        "model": "Cloudflare/clef-flash",
        "state": "The participant asked to activate the journal.",
        "questions": {
            "activate": {
                "type": "choice",
                "instructions": "Did the participant explicitly approve activation?",
                "criteria": {"yes": "Explicit approval", "no": "No explicit approval"},
            }
        },
    }
    return {**body, **overrides}


def test_config_resolves_model_paths_relative_to_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "yaml" / "clef.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(f"model_revision: {REVISION}\nmodel_path: ../weights\nmodel_cache: ../cache\n")

    config = load_config(yaml_path)

    assert config.model_path == (tmp_path / "weights").resolve()
    assert config.model_cache == (tmp_path / "cache").resolve()
    assert config.port == 8120


def test_health_models_and_success_include_revision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        assert client.get("/health").json() == identity(config)
        assert client.get("/v1/models").json()["data"] == [
            {"id": config.model_name, "object": "model", "model_revision": REVISION}
        ]
        response = client.post("/v1/systemone", json=_request())

    assert backend.closed
    assert response.status_code == 200
    assert response.json()["model_revision"] == REVISION
    assert response.json()["answers"]["activate"]["choice"] == "yes"
    assert len(backend.calls) == 2  # startup warmup and the request


def test_empty_instructions_follow_systemone_fallback(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = _Backend(config)
    body = _request()
    body["questions"]["activate"]["instructions"] = ""
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=body)

    assert response.status_code == 200
    assert response.json()["answers"]["activate"]["choice"] == "yes"


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"model": "other"}, "model must be"),
        ({"images": ["unsupported"]}, "text-only"),
        ({"questions": {"activate": {"type": "noul"}}}, "only choice"),
        (
            {"questions": {"activate": {"type": "choice", "instructions": "Choose", "criteria": {"yes": "Yes"}}}},
            "at least two choices",
        ),
    ],
)
def test_invalid_requests_are_rejected_before_inference(tmp_path: Path, overrides: dict, detail: str) -> None:
    config = _config(tmp_path)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(**overrides))

    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert len(backend.calls) == 1  # warmup only


def test_overlong_request_is_rejected_without_truncation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = _Backend(config)
    body = _request(state="too long")
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=body)

    assert response.status_code == 422
    assert "token limit" in response.json()["detail"]
    assert len(backend.calls) == 1


def test_body_limit_is_enforced(tmp_path: Path) -> None:
    config = _config(tmp_path, max_body_bytes=1024)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post(
            "/v1/systemone",
            content=b" " * 1025,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413


def _health_response(config: ServerConfig):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _size: int) -> bytes:
            return json.dumps(identity(config)).encode()


    return _Response()


@pytest.mark.parametrize("host, url_host", [
    ("127.0.0.1", "127.0.0.1"),
    ("0.0.0.0", "127.0.0.1"),
    ("::", "[::1]"),
    ("::1", "[::1]"),
    ("2001:db8::1", "[2001:db8::1]"),
])
def test_matching_owned_ready_listener_is_reused(tmp_path: Path, monkeypatch, host, url_host) -> None:
    config = replace(_config(tmp_path), host=host)
    monkeypatch.setattr(clef_main, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(clef_main, "_has_clef_ownership", lambda *_args: True)
    urls = []

    def probe(url, **_kwargs):
        urls.append(url)
        return _health_response(config)

    monkeypatch.setattr(clef_main, "urlopen", probe)

    assert clef_main._reuse_ready_server(config)
    assert urls == [f"http://{url_host}:{config.port}/health"]


def test_mismatched_configuration_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _size: int) -> bytes:
            mismatched = identity(config) | {"configuration_fingerprint": "other"}
            return json.dumps(mismatched).encode()

    monkeypatch.setattr(clef_main, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(clef_main, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(clef_main, "urlopen", lambda *_args, **_kwargs: _Response())

    with pytest.raises(RuntimeError, match="different Clef configuration"):
        clef_main._reuse_ready_server(config)


def test_unmanaged_listener_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(clef_main, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(clef_main, "_has_clef_ownership", lambda *_args: False)
    monkeypatch.setattr(
        clef_main,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not probe unmanaged listener")),
    )

    with pytest.raises(RuntimeError, match="unmanaged listener"):
        clef_main._reuse_ready_server(config)


def test_listener_replacement_during_reuse_fails_closed(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    listeners = iter([(1234, True, True), (5678, True, True)])
    monkeypatch.setattr(clef_main, "pid_on_port_checked", lambda _port: next(listeners))
    monkeypatch.setattr(clef_main, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(clef_main, "urlopen", lambda *_args, **_kwargs: _health_response(config))

    with pytest.raises(RuntimeError, match="changed during inspection"):
        clef_main._reuse_ready_server(config)


def test_clef_ownership_uses_exact_environment_not_command_line(tmp_path: Path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text("python\0unrelated.py\0/tmp/clef_server.yaml")
    (proc_root / "environ").write_bytes(b"PATH=/bin\0")
    monkeypatch.setattr(
        _docker,
        "Path",
        lambda raw: tmp_path / str(raw).removeprefix("/"),
    )
    assert not clef_main._has_clef_ownership(1234, 8120)
    (proc_root / "environ").write_bytes(
        b"XR_AI_CLEF_MANAGED=1\0XR_AI_CLEF_PORT=8120\0"
    )
    assert clef_main._has_clef_ownership(1234, 8120)
    assert not clef_main._has_clef_ownership(1234, 8121)


@pytest.fixture
def process_handle(monkeypatch):
    monkeypatch.setattr(_docker.os, "pidfd_open", lambda _pid: 999)
    monkeypatch.setattr(_docker.os, "close", lambda _fd: None)


def test_stop_ignores_docker_and_rejects_unowned_listener(monkeypatch, process_handle) -> None:
    monkeypatch.setattr(_docker, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(_docker, "_has_clef_ownership", lambda *_args: False)
    monkeypatch.setattr(
        "xr_ai_vllm._docker.container_on_port_checked",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Clef stop must not inspect Docker")),
    )
    monkeypatch.setattr(
        _docker.signal,
        "pidfd_send_signal",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not signal unowned listener")),
    )

    assert not clef_sample._stop_clef(8120)


def test_stop_rechecks_listener_pid_before_signalling(monkeypatch, process_handle) -> None:
    listeners = iter([(1234, True, True), (5678, True, True)])
    monkeypatch.setattr(_docker, "pid_on_port_checked", lambda _port: next(listeners))
    monkeypatch.setattr(_docker, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(
        _docker.signal,
        "pidfd_send_signal",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not signal replacement listener")),
    )

    assert not clef_sample._stop_clef(8120)


def test_stop_signals_verified_process_handle(monkeypatch, process_handle) -> None:
    signals = []
    monkeypatch.setattr(_docker, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(_docker, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(_docker.signal, "pidfd_send_signal", lambda fd, sig: signals.append((fd, sig)))
    monkeypatch.setattr(_docker, "_wait_for_pidfd_exit", lambda *_args: True)

    assert clef_sample._stop_clef(8120)
    assert signals == [(999, signal.SIGTERM)]


def test_force_stop_keeps_captured_identity_after_listener_replacement(monkeypatch, process_handle) -> None:
    listeners = iter([(1234, True, True), (1234, True, True), (5678, True, True)])
    monkeypatch.setattr(_docker, "pid_on_port_checked", lambda _port: next(listeners))
    monkeypatch.setattr(_docker, "_has_clef_ownership", lambda *_args: True)
    signals = []
    monkeypatch.setattr(_docker.signal, "pidfd_send_signal", lambda fd, sig: signals.append((fd, sig)))
    waits = iter([False, True])
    monkeypatch.setattr(_docker, "_wait_for_pidfd_exit", lambda *_args: next(waits))
    assert clef_sample._stop_clef(8120)
    assert signals == [(999, signal.SIGTERM), (999, signal.SIGKILL)]
    assert next(listeners) == (5678, True, True)


def test_stop_waits_for_real_uvicorn_process_and_native_work(tmp_path, monkeypatch):
    import httpx

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    script = '''
import asyncio, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from clef_server._config import ServerConfig, DEFAULT_REVISION
from clef_server._service import create_app
root=Path(sys.argv[1]);port=int(sys.argv[2])
cfg=ServerConfig('test',DEFAULT_REVISION,None,root,'127.0.0.1',port,'cpu','float32',256,4096)
class Backend:
    model=object();revision=DEFAULT_REVISION;config=cfg
    executor=ThreadPoolExecutor(max_workers=1)
    def validate_length(self,body):pass
    def answer(self,body):
        if body['state']=='blocked':
            (root/'entered').touch()
            while not (root/'release').exists():time.sleep(.01)
        answers={}
        for key,question in body['questions'].items():
            labels=list(question['criteria'])
            answers[key]={'type':'choice','choice':labels[0],'confidence':1,
                          'probabilities':{label:float(i==0) for i,label in enumerate(labels)}}
        return {'answers':answers,'usage':{}}
    async def close(self):
        await asyncio.to_thread(self.executor.shutdown,wait=True)
        (root/'closed').touch()
uvicorn.run(create_app(cfg,Backend()),host='127.0.0.1',port=port,log_level='error')
'''
    env = os.environ | {
        "PYTHONPATH": str(_REPO_ROOT / "services/clef-server"),
        "XR_AI_CLEF_MANAGED": "1", "XR_AI_CLEF_PORT": str(port),
    }
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), str(port)], env=env)
    request_errors = []
    results = []

    def listener(_port):
        with socket.socket() as probe:
            probe.settimeout(0.1)
            return (process.pid, True, True) if probe.connect_ex(("127.0.0.1", port)) == 0 else (None, True, False)

    def request():
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(f"http://127.0.0.1:{port}/v1/systemone", json={
                    "model": "test", "state": "blocked", "questions": {
                        "q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}},
                    },
                })
                response.raise_for_status()
        except BaseException as exc:
            request_errors.append(exc)

    def wait_for(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    request_thread = threading.Thread(target=request)
    stop_thread = threading.Thread(
        target=lambda: results.append(stop_persistent_servers([("clef", port)]))
    )
    try:
        wait_for(lambda: listener(port)[2])
        monkeypatch.setattr(_docker, "pid_on_port_checked", listener)
        request_thread.start()
        wait_for(lambda: (tmp_path / "entered").exists())
        stop_thread.start()
        wait_for(lambda: not listener(port)[2])
        assert process.poll() is None
        assert request_thread.is_alive() and stop_thread.is_alive()
        assert not results and not (tmp_path / "closed").exists()
        (tmp_path / "release").touch()
        request_thread.join(timeout=5)
        stop_thread.join(timeout=5)
        process.wait(timeout=5)
        assert results == [True] and not request_errors
        assert (tmp_path / "closed").exists()
    finally:
        (tmp_path / "release").touch()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if request_thread.ident is not None:
            request_thread.join(timeout=5)
        if stop_thread.ident is not None:
            stop_thread.join(timeout=5)


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_lock_until_worker_completion() -> None:
    lock = asyncio.Lock()
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def work() -> None:
        started.set()
        await release.wait()
        finished.set()

    async def request() -> None:
        async with lock:
            await _await_without_abandoning(asyncio.create_task(work()))

    task = asyncio.create_task(request())
    await started.wait()
    assert lock.locked()

    task.cancel()
    await asyncio.sleep(0)
    assert lock.locked()
    assert not task.done()

    task.cancel()
    await asyncio.sleep(0)
    assert lock.locked()
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    assert not lock.locked()


def test_vendor_rounded_probabilities_are_normalized() -> None:
    criteria = {f"choice-{index}": f"Choice {index}" for index in range(60)}
    request = {
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": criteria,
            }
        }
    }
    result = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": "choice-7",
                "confidence": 0.0167,
                "probabilities": dict.fromkeys(criteria, 0.0167),
            }
        },
        "usage": {},
    }

    normalized = _normalize_response(result, request)
    answer = normalized["answers"]["q"]

    assert math.fsum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert answer["confidence"] == answer["probabilities"]["choice-7"]
    assert answer["confidence"] == pytest.approx(1 / 60)
    assert math.fsum(result["answers"]["q"]["probabilities"].values()) == pytest.approx(1.002)


@pytest.mark.parametrize(
    ("probabilities", "error"),
    [
        ({"yes": True, "no": 0.0}, "finite non-negative"),
        ({"yes": float("nan"), "no": 1.0}, "finite non-negative"),
        ({"yes": float("inf"), "no": 0.0}, "finite non-negative"),
        ({"yes": -0.1, "no": 1.1}, "finite non-negative"),
        ({"yes": 0.0, "no": 0.0}, "positive total"),
    ],
)
def test_malformed_vendor_probabilities_are_rejected(
    probabilities: dict[str, float],
    error: str,
) -> None:
    request = {
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"yes": "Yes", "no": "No"},
            }
        }
    }
    result = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": "yes",
                "confidence": 1.0,
                "probabilities": probabilities,
            }
        }
    }

    with pytest.raises(ValueError, match=error):
        _normalize_response(result, request)


def test_malformed_vendor_response_returns_bad_gateway(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class _MalformedBackend(_Backend):
        def answer(self, body: dict) -> dict:
            if body["state"] == "warmup":
                return super().answer(body)
            return {
                "answers": {
                    "activate": {
                        "type": "choice",
                        "choice": "yes",
                        "confidence": 0.0,
                        "probabilities": {"yes": 0.0, "no": 0.0},
                    }
                },
                "usage": {},
            }

    with TestClient(create_app(config, _MalformedBackend(config))) as client:
        response = client.post("/v1/systemone", json=_request())

    assert response.status_code == 502
    assert "positive total" in response.json()["detail"]
