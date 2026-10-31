# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise procedure image preparation through CLEF's HTTP admission boundary.

Only model inference is scripted; adapter serialization, image decoding, body
and pixel limits, and the sample's fallback policy run as in production.
"""

from __future__ import annotations

import base64
import io
import random
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image
from test_clef_server import _Backend, create_app, load_config
from xr_ai_models import DecisionInputTooLargeError
from xr_ai_models._systemone import _SystemOneDecision
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryResult

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/procedure-capture-guide/worker"))
from procedure_capture_guide_worker.guide.verification import _prepare_decision_image  # noqa: E402
from procedure_capture_guide_worker.guide.visual_check import VisualCheckTrace, verify_image  # noqa: E402


def encoded_image(fmt, size=(1280, 720), *, noise=False):
    pixels = random.Random(19).randbytes(size[0] * size[1] * 3) if noise else bytes([180, 50, 90]) * (size[0] * size[1])
    with Image.frombytes("RGB", size, pixels) as image:
        output = io.BytesIO()
        image.save(output, format=fmt, quality=95)
        return output.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("kind", ["bytes", "path", "inline"])
def test_image_inputs_keep_pixels_when_conversion_fits(tmp_path, fmt, kind):
    raw = encoded_image(fmt, (32, 24))
    source = raw
    if kind == "path":
        source = tmp_path / "frame.bin"
        source.write_bytes(raw)
    elif kind == "inline":
        source = f"data:image/{fmt.lower()};base64," + base64.b64encode(raw).decode()
    prepared = _prepare_decision_image(source)
    with Image.open(io.BytesIO(raw)) as original, Image.open(io.BytesIO(prepared)) as converted:
        assert converted.format in {"JPEG", "PNG"}
        assert converted.size == original.size
        assert converted.convert("RGB").tobytes() == original.convert("RGB").tobytes()
    if fmt != "WEBP":
        assert prepared == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt,size", [("WEBP", (1280, 720)), ("JPEG", (1280, 720)),
                                     ("PNG", (1280, 720)), ("PNG", (3000, 2000))])
async def test_camera_inputs_pass_real_adapter_and_default_server_limits(fmt, size):
    config = load_config(_ROOT / "services/clef-server/clef_server.yaml")
    backend = _Backend(config)
    bodies = []

    async def record_request(request):
        bodies.append(len(request.content))

    query = Mock(execute=AsyncMock())
    original = encoded_image(fmt, size, noise=True)
    frame = {"participant_id": "alice", "track": "camera", "sequence": 42, "timestamp_us": 1234}
    trace = VisualCheckTrace()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(config, backend)),
                                     event_hooks={"request": [record_request]}) as client:
            decisions = _SystemOneDecision(base_url="http://clef.test", model_name=config.model_name,
                                           api_key_env=None, timeout=5, health_check=False,
                                           health_path="/health", client=client)
            if fmt == "JPEG":
                # Reproduce the old path's admission failure with the hub's
                # quality-95 JPEG and real serialization, before preparation.
                from procedure_capture_guide_worker.guide.verification import _IMAGE_QUESTION
                with pytest.raises(DecisionInputTooLargeError, match="body"):
                    await decisions.decide("condition", {"condition_met": _IMAGE_QUESTION}, images=(original,))
                assert bodies[-1] > config.max_body_bytes
            result = await verify_image(
                decisions, query, images=ImageRegistry(), image=original,
                condition="The display reads 100 C.", question="Describe the display digits and unit.",
                frame=frame, step_id="heat", revision=2, decision_timeout_s=5, vision_timeout_s=5, trace=trace,
            )
    finally:
        await backend.close()
    assert result.choice == "yes" and trace.source == "image"
    query.execute.assert_not_awaited()
    assert len(backend.calls) == 1 and bodies[-1] < config.max_body_bytes
    assert frame == {"participant_id": "alice", "track": "camera", "sequence": 42, "timestamp_us": 1234}
    admitted = backend.calls[0]["images"][0]
    assert admitted.width * admitted.height <= 2_000_000
    assert abs(admitted.width / admitted.height - size[0] / size[1]) < 0.01


@pytest.mark.asyncio
async def test_low_confidence_uses_original_webp_for_caption():
    from _procedure_guide_helpers import clef_response

    original = encoded_image("WEBP", noise=True)
    registry = ImageRegistry()
    decisions = Mock(decide=AsyncMock(side_effect=[clef_response("no", .8), clef_response("yes")]))

    async def describe(request):
        assert registry.resolve(request.image) == original
        return ImageQueryResult(text="The display reads 100 C.")

    trace = VisualCheckTrace()
    await verify_image(decisions, Mock(execute=AsyncMock(side_effect=describe)), images=registry, image=original,
                       condition="100 C", question="Read display.", frame={"sequence": 42}, step_id="heat",
                       revision=1, decision_timeout_s=5, vision_timeout_s=5, trace=trace)
    assert trace.source == "caption" and trace.answer.choice == "yes"
    prepared = decisions.decide.await_args_list[0].kwargs["images"][0]
    assert prepared != original and len(prepared) <= 512 * 1024
    assert "images" not in decisions.decide.await_args_list[1].kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["body", "pixels", "invalid"])
async def test_admission_failure_never_becomes_low_confidence(failure):
    config = load_config(_ROOT / "services/clef-server/clef_server.yaml")
    if failure == "body":
        config = replace(config, max_body_bytes=1024)
    elif failure == "pixels":
        config = replace(config, max_length=256)
    backend = _Backend(config)
    query = Mock(execute=AsyncMock())
    trace = VisualCheckTrace()
    original = b"not an image" if failure == "invalid" else encoded_image("JPEG", noise=True)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(config, backend))) as client:
            decisions = _SystemOneDecision(base_url="http://clef.test", model_name=config.model_name,
                                           api_key_env=None, timeout=5, health_check=False,
                                           health_path="/health", client=client)
            error = Image.UnidentifiedImageError if failure == "invalid" else DecisionInputTooLargeError
            with pytest.raises(error):
                await verify_image(decisions, query, images=ImageRegistry(), image=original,
                                   condition="100 C", question="Read display.", frame={"sequence": 42},
                                   step_id="heat", revision=1, decision_timeout_s=5, vision_timeout_s=5, trace=trace)
    finally:
        await backend.close()
    query.execute.assert_not_awaited()
    assert not backend.calls and trace.source == "image" and trace.image_answer is None and trace.answer is None
