# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and config contract tests for the local Clef model server."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from xr_ai_launcher import _ownership, has_ownership_marker

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "services" / "clef-server"))

import clef_server.__main__ as clef_main  # noqa: E402
from clef_server._config import ServerConfig, identity, load_config  # noqa: E402
from clef_server._images import _ImageLimits  # noqa: E402
from clef_server._service import (  # noqa: E402
    ClefBackend,
    InputTooLong,
    _await_without_abandoning,
    _normalize_response,
    create_app,
)

REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"


class _Backend:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.revision = REVISION
        self.model = object()
        self.image_limits = _ImageLimits.from_processor(SimpleNamespace(image_processor=SimpleNamespace(
            patch_size=16, merge_size=2, size={"shortest_edge": 65536},
        )))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.calls: list[dict] = []
        self.closed = False

    def validate_length(self, body: dict) -> None:
        if body["state"] == "too long":
            raise InputTooLong("request exceeds the configured token limit")

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


def test_config_resolves_model_cache_relative_to_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "yaml" / "clef.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(f"model_revision: {REVISION}\nmodel_cache: ../cache\n")

    config = load_config(yaml_path)

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
        ({"images": ["unsupported"]}, "inline JPEG"),
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

    assert response.status_code == 413
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
    monkeypatch.setattr(clef_main, "has_ownership_marker", lambda *_args: True)
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
    monkeypatch.setattr(clef_main, "has_ownership_marker", lambda *_args: True)
    monkeypatch.setattr(clef_main, "urlopen", lambda *_args, **_kwargs: _Response())

    with pytest.raises(RuntimeError, match="different Clef configuration"):
        clef_main._reuse_ready_server(config)


def test_unmanaged_listener_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(clef_main, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(clef_main, "has_ownership_marker", lambda *_args: False)
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
    monkeypatch.setattr(clef_main, "has_ownership_marker", lambda *_args: True)
    monkeypatch.setattr(clef_main, "urlopen", lambda *_args, **_kwargs: _health_response(config))

    with pytest.raises(RuntimeError, match="changed during inspection"):
        clef_main._reuse_ready_server(config)


def test_clef_ownership_uses_exact_environment_not_command_line(tmp_path: Path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text("python\0unrelated.py\0/tmp/clef_server.yaml")
    (proc_root / "environ").write_bytes(b"PATH=/bin\0")
    monkeypatch.setattr(
        _ownership,
        "Path",
        lambda raw: tmp_path / str(raw).removeprefix("/"),
    )

    assert not has_ownership_marker(1234, 8120, clef_main._MARKER)
    (proc_root / "environ").write_bytes(
        b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8120\0"
    )
    assert has_ownership_marker(1234, 8120, clef_main._MARKER)
    assert not has_ownership_marker(1234, 8121, clef_main._MARKER)


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


def _image_url(mode="RGB", size=(8, 8), format="PNG"):
    import base64
    import io

    from PIL import Image

    with Image.new(mode, size, color=0) as image, io.BytesIO() as buffer:
        image.save(buffer, format=format)
        mime = "image/jpeg" if format == "JPEG" else "image/png"
        return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("format,mode", [("PNG", "RGBA"), ("JPEG", "RGB")])
def test_images_reach_native_backend_in_order_as_rgb(tmp_path, format, mode):
    from PIL import Image

    config = _config(tmp_path, max_body_bytes=4096)
    observed = []

    class Backend(_Backend):
        def validate_length(self, body):
            for image in body.get("images", []):
                assert isinstance(image, Image.Image)
                assert image.mode == "RGB"
                observed.append(image.size)
            super().validate_length(body)

        def answer(self, body):
            for image in body.get("images", []):
                assert image.getpixel((0, 0)) == (0, 0, 0)
            return super().answer(body)

    backend = Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(images=[
            _image_url(mode=mode, format=format), _image_url(size=(9, 7)),
        ]))
    assert response.status_code == 200
    assert observed == [(8, 8), (9, 7)]
    # Native images are released even though the fake retains its request.
    with pytest.raises(ValueError):
        backend.calls[-1]["images"][0].getpixel((0, 0))


@pytest.mark.parametrize("images", [
    None, {}, "data:image/png;base64,abc", ["https://example.com/photo.png"],
    ["/tmp/photo.png"], ["data:image/gif;base64,R0lGODlh"],
    ["data:image/png;base64,%%%%"], ["data:image/png;base64,YWJj"],
    [_image_url(format="JPEG").replace("image/jpeg", "image/png")],
    [_image_url(), "data:image/png;base64,YWJj"],
])
def test_bad_images_fail_before_native_inference(tmp_path, images):
    config = _config(tmp_path, max_body_bytes=4096)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(images=images))
    assert response.status_code == 422
    assert len(backend.calls) == 1


def test_aggregate_source_pixels_rejected_before_rgb_decode(tmp_path, monkeypatch):
    from PIL import Image

    config = replace(_config(tmp_path, max_body_bytes=16384), max_length=256)
    # Each image fits 256*32*32; together they exceed that budget.
    image = _image_url(size=(400, 400))
    original = Image.Image.convert
    converted = []

    def convert(source, *args, **kwargs):
        converted.append(source.size)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", convert)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(images=[image, image]))
    assert response.status_code == 422
    assert "source-pixel budget" in response.json()["detail"]
    assert converted == [(400, 400)]
    assert len(backend.calls) == 1


@pytest.mark.parametrize("failure", ["validate_length", "answer"])
def test_image_native_failure_releases_images(tmp_path, failure):
    config = _config(tmp_path)
    retained = []

    class Backend(_Backend):
        def validate_length(self, body):
            retained.extend(body.get("images", []))
            if failure == "validate_length":
                raise ValueError("too many tokens")

        def answer(self, body):
            if failure == "answer" and body["state"] != "warmup":
                raise ValueError("native answer failed")
            return super().answer(body)

    backend = Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(images=[_image_url(), _image_url()]))
    assert response.status_code == 422
    assert len(backend.calls) == 1
    assert len(retained) == 2
    for image in retained:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


def _broken_png_url():
    import base64
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xffffffff)

    raw = base64.b64decode(_image_url().split(",", 1)[1])
    index = raw.index(b"IDAT")
    length = struct.unpack("!I", raw[index - 4:index])[0]
    broken = raw[:index - 4] + chunk(b"IDAT", raw[index + 4:index + 4 + length // 2])
    broken += chunk(b"\x00\x01\x02\x03", b"") + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(broken).decode()


@pytest.mark.parametrize("error", [None, RuntimeError("unexpected conversion failure")])
def test_partial_image_decode_failure_always_releases_images(tmp_path, monkeypatch, error):
    from PIL import Image

    converted = []
    original = Image.Image.convert

    def convert(source, *args, **kwargs):
        if converted and error is not None:
            raise error
        image = original(source, *args, **kwargs)
        converted.append(image)
        return image

    monkeypatch.setattr(Image.Image, "convert", convert)
    config = _config(tmp_path, max_body_bytes=4096)
    backend = _Backend(config)
    with TestClient(create_app(config, backend), raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", json=_request(images=[
            _image_url(), _broken_png_url() if error is None else _image_url(),
        ]))
    assert response.status_code == (422 if error is None else 500)
    if error is None:
        assert "broken PNG" in response.json()["detail"]
    assert len(backend.calls) == 1
    assert len(converted) == 1
    with pytest.raises(ValueError, match="closed image"):
        converted[0].getpixel((0, 0))


@pytest.mark.parametrize("options", [None, {}, {
    "images_kwargs": {"size": {"shortest_edge": 1073741824, "longest_edge": 1073741824}},
}])
def test_processor_overrides_rejected_before_decode_or_native_work(tmp_path, monkeypatch, options):
    import clef_server._service as service

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported preprocessing options reached image decoding")

    monkeypatch.setattr(service, "decode_images", unexpected)
    config = _config(tmp_path, max_body_bytes=4096)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(images=[_image_url()], media_kwargs=options))
    assert response.status_code == 422
    assert "server-owned" in response.json()["detail"]
    assert len(backend.calls) == 1


def test_native_body_contains_only_the_service_owned_contract(tmp_path):
    config = _config(tmp_path)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(future_native_option={"enabled": True}))
    assert response.status_code == 200
    assert backend.calls[-1].keys() == {"model", "state", "questions"}


def test_image_admission_uses_loaded_processor_geometry(tmp_path, monkeypatch):
    from PIL import Image

    config = replace(_config(tmp_path, max_body_bytes=16384), max_length=256)
    backend = _Backend(config)
    backend.image_limits = _ImageLimits.from_processor(SimpleNamespace(image_processor=SimpleNamespace(
        patch_size=8, merge_size=2, size={"shortest_edge": 32768},
    )))
    converted = []
    original = Image.Image.convert

    def convert(source, *args, **kwargs):
        converted.append(source.size)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", convert)
    with TestClient(create_app(config, backend)) as client:
        # This fits the old pinned geometry, but exceeds the actual pixel budget.
        response = client.post("/v1/systemone", json=_request(images=[_image_url(size=(300, 300))]))
        assert response.status_code == 422
        assert "source-pixel budget" in response.json()["detail"]
        # The processor minimum is now 128 tokens per image, not 64.
        response = client.post("/v1/systemone", json=_request(images=[_image_url()] * 3))
        assert response.status_code == 422
        assert "too many images" in response.json()["detail"]
    assert converted == []
    assert len(backend.calls) == 1


@pytest.mark.parametrize("actual_length", [32767, 32768, 32769, 40000])
def test_length_admission_detects_native_truncation(tmp_path, actual_length):
    config = replace(_config(tmp_path), max_length=32768)
    backend = ClefBackend(config)
    backend.processor = SimpleNamespace(tokenizer=object())
    caps = []

    def encode(tokenizer, body, *, max_length, processor):
        # The native encoder truncates state to fit its cap after adding media
        # and schema tokens. Exercise both sides of our own admission boundary.
        caps.append(max_length)
        return SimpleNamespace(input_ids=range(min(actual_length, max_length)))

    backend.encode_record = encode
    try:
        if actual_length > config.max_length:
            with pytest.raises(InputTooLong, match="maximum is 32768"):
                backend.validate_length(_request(images=[_image_url()]))
        else:
            backend.validate_length(_request(images=[_image_url()]))
        assert caps == [config.max_length + 1]
    finally:
        backend.executor.shutdown(wait=True)


def test_videos_still_rejected(tmp_path):
    config = _config(tmp_path)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(videos=[]))
    assert response.status_code == 422
    assert "videos" in response.json()["detail"]


@pytest.mark.asyncio
async def test_typed_image_client_round_trip_to_server(tmp_path):
    import base64

    import httpx
    from xr_ai_models import DecisionQuestion
    from xr_ai_models._systemone import _SystemOneDecision

    config = _config(tmp_path, max_body_bytes=4096)
    backend = _Backend(config)
    transport = httpx.ASGITransport(app=create_app(config, backend))
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            service = _SystemOneDecision(
                base_url="http://clef.test", model_name=config.model_name, api_key_env=None,
                timeout=9, health_check=False, health_path="/health", client=client,
            )
            raw_image = base64.b64decode(_image_url().split(",", 1)[1])
            result = await service.decide("target image", {
                "q": DecisionQuestion(instructions="Choose", criteria={"yes": "Yes", "no": "No"}),
            }, images=[raw_image])
            assert result.answers["q"].choice == "yes"
            assert result.raw["model_revision"] == REVISION
    finally:
        await backend.close()
    assert len(backend.calls) == 1
    assert backend.calls[0]["images"][0].mode == "RGB"


def test_image_count_budget_accounts_for_processor_upscaling(tmp_path):
    config = replace(_config(tmp_path, max_body_bytes=4096), max_length=256)
    backend = _Backend(config)
    with TestClient(create_app(config, backend)) as client:
        response = client.post("/v1/systemone", json=_request(images=[_image_url()] * 5))
    assert response.status_code == 422
    assert "too many images" in response.json()["detail"]
    assert len(backend.calls) == 1
