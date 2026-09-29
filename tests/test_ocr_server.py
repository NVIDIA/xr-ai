# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCR protocol, bounded local inference, and deployment tests (no GPU)."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont
from xr_ai_models import OCRService, OCRSpec, load_models_config, load_models_config_from_dict, make_ocr
from xr_ai_models._ocr import NemotronOCR

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ocr_server_app", ROOT / "services/ocr-server/ocr_server/server.py")
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


def image_url(size=(16, 12)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def response():
    return {"data": [{"index": 0, "text_detections": [{
        "text_prediction": {"text": "温度 23.5", "confidence": 0.95},
        "bounding_box": {"points": [
            {"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2},
            {"x": 0.9, "y": 0.4}, {"x": 0.1, "y": 0.4},
        ]},
    }]}]}


def config():
    return load_models_config_from_dict({"ocr": {
        "adapter": {"preset": "nemotron_ocr"}, "endpoint": {"base_url": "http://localhost:8112"},
    }})


def test_inline_ocr_defaults_to_its_native_readiness_route():
    models = load_models_config_from_dict({"ocr": {
        "category": "ocr", "adapter": {"kind": "nemotron_ocr"},
        "endpoint": {"base_url": "http://localhost:8112"},
    }})
    assert models.ocr("ocr").health_path == "/v1/health/ready"


async def test_sdk_contract_and_lifecycle():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"ready": True} if request.method == "GET" else response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        ocr = NemotronOCR(config().ocr("ocr"), client=client)
        assert isinstance(ocr, OCRService)
        assert await ocr.health()
        spans = await ocr.read_text(image_url(), merge_level="paragraph", timeout=10)
        assert spans[0].text == "温度 23.5"
        assert spans[0].polygon[0] == (0.1, 0.2)
        assert spans[0].confidence == 0.95
        assert calls[0].url.path == "/v1/health/ready"
        assert calls[1].url.path == "/v1/ocr"
        assert json.loads(calls[1].content)["merge_levels"] == ["paragraph"]
        await ocr.close()
        assert not client.is_closed
        with pytest.raises(ValueError, match="inline"):
            await ocr.read_text("http://internal/image")
        with pytest.raises(ValueError, match="merge_level"):
            await ocr.read_text(image_url(), merge_level="invalid")
    await make_ocr(config(), "ocr").close()
    with pytest.raises(TypeError):
        config().llm("ocr")


@pytest.mark.parametrize("kind,category", [("openai_compat", "ocr"), ("nemotron_ocr", "stt")])
def test_reject_wrong_adapter_category(kind, category):
    with pytest.raises(ValueError, match="ocr requires"):
        load_models_config_from_dict({"bad": {"category": category, "kind": kind, "base_url": "http://localhost"}})


@pytest.mark.parametrize("hardware,arch,gpu", [
    ("96G_blackwell", "12.0", "0"), ("dual_48G_ada", "8.9", "1"), ("spark", "12.1", "0"),
])
async def test_hardware_profiles(hardware, arch, gpu):
    local = ROOT / "model-server-samples/model-servers/yaml"
    nim = ROOT / "model-server-samples/model-servers-nim/yaml" / hardware
    hf = yaml.safe_load((local / hardware / "ocr_server.yaml").read_text())
    assert hf["cuda_arch"] == arch
    assert hf["cuda_visible_devices"] == "0"
    assert hf["infer_length"] == 1024
    cfg = yaml.safe_load((nim / "nim_ocr_server.yaml").read_text())
    assert cfg["cuda_visible_devices"] == gpu
    assert cfg["http_port"] == hf["port"] == 8112
    assert "@sha256:" in cfg["image"]
    assert cfg["env"]["NIM_ENGINE_MODEL_VARIANT"] == "multilingual"
    assert cfg["env"]["NIM_ENGINE_COUNT"] == cfg["env"]["NIM_PIPELINE_MAX_BATCH_SIZE"] == "1"
    assert cfg["env"]["NIM_ENGINE_MODEL_PATH"].startswith("/opt/nim/.cache/")
    for path in [local / "models.default.json", local / "models.vlm_llm_nim.json", nim / "models.json"]:
        models = load_models_config(path)
        assert isinstance(models.ocr("ocr"), OCRSpec)
        await make_ocr(models, "ocr").close()


@pytest.mark.parametrize("url", [
    "http://localhost/private", "file:///etc/passwd", "data:image/png;base64,not-base64",
    "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
])
def test_reject_invalid_images(url):
    with pytest.raises(HTTPException):
        server.decode_image(url)


def test_image_limits(monkeypatch):
    monkeypatch.setattr(server, "MAX_PADDED_IMAGE_PIXELS", 10)
    with pytest.raises(HTTPException) as exc:
        server.decode_image(image_url())
    assert exc.value.status_code == 413


@pytest.mark.parametrize("size", [(100_000, 1), (1, 100_000), (4473, 1), (1, 4473)])
async def test_thin_image_rejected_before_conversion_or_model(size, monkeypatch):
    url = image_url(size)
    # These inputs satisfy the old byte-count and decoded-area checks.
    assert len(base64.b64decode(url.partition(",")[2])) < server.MAX_IMAGE_BYTES
    assert size[0] * size[1] < server.MAX_PADDED_IMAGE_PIXELS

    def unexpected_call(*args, **kwargs):
        pytest.fail("oversized padded image reached conversion or model inference")

    monkeypatch.setattr(Image.Image, "convert", unexpected_call)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=unexpected_call))
    app = server.build_app(loader=lambda: unexpected_call)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            result = await client.post("/v1/ocr", json={"input": [{"type": "image_url", "url": url}]})
    assert result.status_code == 413
    assert "padded-square" in result.json()["detail"]


@pytest.mark.parametrize("size", [(4472, 1), (1, 4472), (1920, 1080)])
def test_images_within_padded_limit_are_accepted(size):
    with server.decode_image(image_url(size)) as image:
        assert image.size == size
        assert image.mode == "RGB"


def test_exact_padded_area_boundary(monkeypatch):
    monkeypatch.setattr(server, "MAX_PADDED_IMAGE_PIXELS", 16 ** 2)
    with server.decode_image(image_url((16, 16))) as image:
        assert image.size == (16, 16)
    with pytest.raises(HTTPException) as exc:
        server.decode_image(image_url((17, 1)))
    assert exc.value.status_code == 413


def test_normalizes_upstream_inverted_y(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    def model(image, merge_level):
        assert image.shape == (12, 16, 3) and merge_level == "word"
        return [{"text": "ABC", "confidence": 0.9, "left": .1, "right": .8, "upper": .7, "lower": .2}]
    result = server.transcribe(model, server.OCRRequest(input=[{"type": "image_url", "url": image_url()}]))
    points = result["data"][0]["text_detections"][0]["bounding_box"]["points"]
    assert points[0] == {"x": .1, "y": .2} and points[2] == {"x": .8, "y": .7}


async def test_app_readiness_validation_and_inference():
    app = server.build_app(lambda: object(), lambda model, req: response())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.get("/v1/health/ready")).status_code == 503
        async with app.router.lifespan_context(app):
            assert (await client.get("/v1/health/ready")).status_code == 200
            item = {"type": "image_url", "url": image_url()}
            assert (await client.post("/v1/ocr", json={"input": [item]})).json() == response()
            for payload in [{"input": []}, {"input": [item, item]}, {"input": [item], "merge_levels": ["invalid"]}]:
                assert (await client.post("/v1/ocr", json=payload)).status_code == 422


async def test_busy_and_cancelled_requests_do_not_overlap():
    started, release = threading.Event(), threading.Event()
    def inference(model, req):
        started.set()
        assert release.wait(5)
        return response()
    app = server.build_app(lambda: object(), inference)
    payload = {"input": [{"type": "image_url", "url": image_url()}]}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/ocr", json=payload))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                first.cancel()
                await asyncio.sleep(0)
                assert (await client.post("/v1/ocr", json=payload)).status_code == 503
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            async with asyncio.timeout(2):
                while (await client.post("/v1/ocr", json=payload)).status_code == 503:
                    await asyncio.sleep(0.01)


def test_local_loader_pins_multilingual_and_bounds_chunks(monkeypatch):
    downloads, options = [], []
    class Model:
        def __init__(self, **kwargs):
            options.append(kwargs)
        def __call__(self, image, **kwargs):
            assert image.shape == (640, 640, 3)
    monkeypatch.setenv("OCR_INFER_LENGTH", "640")
    monkeypatch.setenv("OCR_MODEL_REVISION", "pinned-revision")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
        set_num_threads=lambda value: None,
    ))
    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return "/cache/snapshot"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "nemotron_ocr.inference.pipeline_v2", SimpleNamespace(NemotronOCRV2=Model))
    server.load_model()
    assert downloads == [((server.MODEL_ID,), {
        "revision": "pinned-revision", "allow_patterns": ["v2_multilingual/*"],
    })]
    assert options == [{
        "model_dir": "/cache/snapshot/v2_multilingual", "lang": "multi", "infer_length": 640,
        "detector_max_batch_size": 1, "recognizer_chunk_size": 32, "relational_chunk_size": 32,
    }]


def test_local_launcher_uses_shared_lifecycle_and_image_fingerprint(monkeypatch, tmp_path):
    import_spec = importlib.util.spec_from_file_location(
        "ocr_launcher", ROOT / "services/ocr-server/ocr_server/__main__.py",
    )
    launcher = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(launcher)
    cfg = {"port": 8112, "infer_length": 640, "cuda_arch": "8.9", "cuda_visible_devices": "1",
           "model_cache": "cache", "container_name": "xr-ai-ocr-test"}
    monkeypatch.setattr(launcher, "setup_logging", lambda name: None)
    monkeypatch.setattr(launcher, "load_config", lambda: (cfg, tmp_path, tmp_path / "ready"))
    builds, runs = [], []
    monkeypatch.setattr(launcher.subprocess, "run", lambda args, **kw: builds.append(args))
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **kw: "sha256:first")
    monkeypatch.setattr(launcher._docker, "run_container", lambda **kw: runs.append(kw))
    monkeypatch.setenv("HF_TOKEN", "private-test-token")
    launcher.run()
    assert "TORCH_CUDA_ARCH_LIST=8.9" in builds[0]
    assert "--provenance=false" in builds[0]
    assert runs[0]["ready_file"] == tmp_path / "ready"
    assert "OCR_INFER_LENGTH=640" in runs[0]["argv"]
    assert "NVIDIA_VISIBLE_DEVICES=1" in runs[0]["argv"]
    assert "HF_TOKEN" in runs[0]["argv"] and "private-test-token" not in str(runs)
    assert f"{tmp_path / 'cache'}:/models" in runs[0]["argv"]
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **kw: "sha256:second")
    launcher.run()
    assert runs[0]["argv"] != runs[1]["argv"]


@pytest.mark.gpu
@pytest.mark.parametrize("merge_level", ["word", "sentence", "paragraph"])
async def test_live_ocr_equipment_text(merge_level):
    """Opt-in smoke test: set XR_AI_OCR_TEST_URL to a running OCR endpoint."""
    url = os.environ.get("XR_AI_OCR_TEST_URL")
    if not url:
        pytest.skip("set XR_AI_OCR_TEST_URL to the OCR server under test")
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.text((80, 220), "TEMPERATURE 23.5", font=font, fill="black")
    draw.text((80, 300), "PRESSURE 120", font=font, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    models = load_models_config_from_dict({"ocr": {
        "adapter": {"preset": "nemotron_ocr"}, "endpoint": {"base_url": url},
    }})
    async with make_ocr(models, "ocr") as client:
        assert await client.health()
        spans = await client.read_text(buffer.getvalue(), merge_level=merge_level)
    text = " ".join(span.text for span in spans)
    assert all(token in text for token in ["TEMPERATURE", "23.5", "PRESSURE", "120"])
