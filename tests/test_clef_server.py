# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and config contract tests for the local Clef model server."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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


def test_matching_owned_ready_listener_is_reused(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(clef_main, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(clef_main, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(clef_main, "urlopen", lambda *_args, **_kwargs: _health_response(config))

    assert clef_main._reuse_ready_server(config)


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
        clef_main,
        "Path",
        lambda raw: tmp_path / str(raw).removeprefix("/"),
    )
    monkeypatch.setattr(
        clef_sample,
        "Path",
        lambda raw: tmp_path / str(raw).removeprefix("/"),
    )

    assert not clef_main._has_clef_ownership(1234, 8120)
    assert not clef_sample._has_clef_ownership(1234, 8120)
    (proc_root / "environ").write_bytes(
        b"XR_AI_CLEF_MANAGED=1\0XR_AI_CLEF_PORT=8120\0"
    )
    assert clef_main._has_clef_ownership(1234, 8120)
    assert clef_sample._has_clef_ownership(1234, 8120)
    assert not clef_main._has_clef_ownership(1234, 8121)
    assert not clef_sample._has_clef_ownership(1234, 8121)


def test_stop_ignores_docker_and_rejects_unowned_listener(monkeypatch) -> None:
    monkeypatch.setattr(clef_sample, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(clef_sample, "_has_clef_ownership", lambda *_args: False)
    monkeypatch.setattr(
        "xr_ai_vllm._docker.container_on_port_checked",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Clef stop must not inspect Docker")),
    )
    monkeypatch.setattr(
        clef_sample.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not signal unowned listener")),
    )

    assert not clef_sample._stop_clef(8120)


def test_stop_rechecks_listener_pid_before_signalling(monkeypatch) -> None:
    listeners = iter([(1234, True, True), (5678, True, True)])
    monkeypatch.setattr(clef_sample, "pid_on_port_checked", lambda _port: next(listeners))
    monkeypatch.setattr(clef_sample, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(
        clef_sample.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not signal replacement listener")),
    )

    assert not clef_sample._stop_clef(8120)


def test_stop_signals_verified_listener_pid(monkeypatch) -> None:
    listeners = iter(
        [
            (1234, True, True),
            (1234, True, True),
            (None, True, False),
        ]
    )
    signals = []
    monkeypatch.setattr(clef_sample, "pid_on_port_checked", lambda _port: next(listeners))
    monkeypatch.setattr(clef_sample, "_has_clef_ownership", lambda *_args: True)
    monkeypatch.setattr(clef_sample.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    assert clef_sample._stop_clef(8120)
    assert signals == [(1234, signal.SIGTERM)]


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
