# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and config contract tests for the local Clef model server."""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "services" / "clef-server"))

from clef_server.__main__ import _reuse_ready_server  # noqa: E402
from clef_server._config import ServerConfig, load_config  # noqa: E402
from clef_server._service import _await_without_abandoning, create_app  # noqa: E402

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
        return {
            "model": body["model"],
            "answers": {
                key: {"type": "choice", "choice": "yes", "confidence": 0.9, "probabilities": {"yes": 0.9, "no": 0.1}}
                for key in body["questions"]
            },
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
        assert client.get("/health").json() == {
            "status": "ready",
            "model": config.model_name,
            "model_revision": REVISION,
        }
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


def test_matching_ready_listener_is_reused(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _size: int) -> bytes:
            return f'{{"status":"ready","model":"Cloudflare/clef-flash","model_revision":"{REVISION}"}}'.encode()

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr("clef_server.__main__.socket.create_connection", lambda *_args, **_kwargs: _Connection())
    monkeypatch.setattr("clef_server.__main__.urlopen", lambda *_args, **_kwargs: _Response())

    assert _reuse_ready_server("127.0.0.1", 8120, "Cloudflare/clef-flash", REVISION)


def test_mismatched_ready_listener_is_not_reused(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _size: int) -> bytes:
            return b'{"status":"ready","model":"other","model_revision":"other"}'

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr("clef_server.__main__.socket.create_connection", lambda *_args, **_kwargs: _Connection())
    monkeypatch.setattr("clef_server.__main__.urlopen", lambda *_args, **_kwargs: _Response())

    with pytest.raises(RuntimeError, match="does not match"):
        _reuse_ready_server("127.0.0.1", 8120, "Cloudflare/clef-flash", REVISION)


@pytest.mark.asyncio
async def test_cancelled_inference_waits_for_worker_completion() -> None:
    finished = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.02)
        finished.set()

    task = asyncio.create_task(_await_without_abandoning(asyncio.create_task(work())))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
