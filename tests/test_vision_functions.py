# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the native image-question vision function."""

from __future__ import annotations

import base64
import time
from types import SimpleNamespace

import httpx
import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from PIL import Image
from pydantic import ValidationError
from xr_ai_agent import FrameData, FrameSignal, PixelFormat
from xr_ai_nat.functions.vision import (
    StreamingVisionConfig,
    VisionFunctionsConfig,
    VisionRequest,
)
from xr_ai_nat.functions.vision._images import load_jpeg_data_url


class _Vlm:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[str, str, str]] = []

    async def ask_image(self, image: str, question: str, *, system_prompt: str = ""):
        self.calls.append((image, question, system_prompt))
        return SimpleNamespace(content=self.content)

    async def stream(self, image: str, question: str, *, system_prompt: str = ""):
        self.calls.append((image, question, system_prompt))
        for token in ("a ", "blue ", "square"):
            yield token


class _Endpoint:
    def __init__(self) -> None:
        self.frame_callback = None
        self.statuses: list[tuple[str, str]] = []

    def on_frame(self, callback) -> None:
        self.frame_callback = callback

    def on_participant(self, _callback) -> None:
        pass

    async def request_frame(self, signal: FrameSignal) -> FrameData:
        return FrameData(
            seq=signal.seq,
            pts_us=signal.pts_us,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data=bytes([20, 40, 60] * 4),
            participant_id=signal.participant_id,
            track_id=signal.track_id,
        )

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))


class _HttpErrorVlm(_Vlm):
    async def ask_image(self, image: str, question: str, *, system_prompt: str = ""):
        raise httpx.HTTPError("backend unavailable")


def test_load_jpeg_data_url_emits_data_url(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_path)

    image_url = load_jpeg_data_url(image_path)

    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\xff\xd8")


async def test_vision_function_normalizes_image_and_returns_clean_answer(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_path)
    vlm = _Vlm("<think>inspect pixels</think>\n  a blue square  ")
    config = VisionFunctionsConfig(vlm=vlm, system_prompt="Answer briefly.")

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("vision", config)
        group = await builder.get_function_group("vision")
        functions = await group.get_all_functions()
        answer = await functions["vision__ask_image"].ainvoke(
            {"question": "What is shown?", "image_path": str(image_path)}
        )

    assert set(functions) == {"vision__ask_image"}
    assert set(functions["vision__ask_image"].input_schema.model_json_schema()["properties"]) == {
        "question",
        "image_path",
    }
    assert answer == "a blue square"
    image_url, question, system_prompt = vlm.calls[0]
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\xff\xd8")
    dumped_config = config.model_dump()
    assert "vlm" not in dumped_config
    assert dumped_config["system_prompt"] == "Answer briefly."


async def test_vision_function_reports_missing_image_without_calling_model(tmp_path) -> None:
    vlm = _Vlm("unused")
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("vision", VisionFunctionsConfig(vlm=vlm))
        group = await builder.get_function_group("vision")
        functions = await group.get_all_functions()
        answer = await functions["vision__ask_image"].ainvoke(
            {"question": "What is shown?", "image_path": str(tmp_path / "missing.png")}
        )

    assert "file not found" in answer
    assert vlm.calls == []


async def test_vision_function_reports_empty_image_path_without_calling_model() -> None:
    vlm = _Vlm("unused")
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("vision", VisionFunctionsConfig(vlm=vlm))
        group = await builder.get_function_group("vision")
        functions = await group.get_all_functions()
        answer = await functions["vision__ask_image"].ainvoke(
            {"question": "What is shown?", "image_path": ""}
        )

    assert answer == "ask_image: image_path is empty — acquire an image first."
    assert vlm.calls == []


async def test_vision_function_reports_http_error(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_path)
    vlm = _HttpErrorVlm("unused")
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("vision", VisionFunctionsConfig(vlm=vlm))
        group = await builder.get_function_group("vision")
        functions = await group.get_all_functions()
        answer = await functions["vision__ask_image"].ainvoke(
            {"question": "What is shown?", "image_path": str(image_path)}
        )

    assert answer == "ask_image: vlm-server request failed: backend unavailable"


async def test_streaming_vision_function_uses_current_participant_frame() -> None:
    endpoint = _Endpoint()
    vlm = _Vlm("a blue square")
    config = StreamingVisionConfig(
        endpoint=endpoint,
        vlm=vlm,
        system_prompt="Answer briefly.",
    )

    async with WorkflowBuilder() as builder:
        function = await builder.add_function("perception", config)
        assert endpoint.frame_callback is not None
        await endpoint.frame_callback(
            FrameSignal(
                slot=0,
                seq=1,
                pts_us=time.time_ns() // 1_000,
                width=2,
                height=2,
                fmt=PixelFormat.RGB24,
                data_sz=12,
                participant_id="alice",
                track_id="camera",
            )
        )
        chunks = [
            chunk.text
            async for chunk in function.astream(VisionRequest(participant_id="alice", query="What is shown?"))
        ]
        answer = await function.ainvoke(
            VisionRequest(participant_id="alice", query="What is shown?")
        )

    assert chunks == ["a ", "blue ", "square"]
    assert answer.text == "a blue square"
    assert endpoint.statuses == [
        ("processing", "alice"),
        ("idle", "alice"),
        ("processing", "alice"),
        ("idle", "alice"),
    ]
    assert vlm.calls[0][1:] == ("What is shown?", "Answer briefly.")
    assert vlm.calls[0][0].startswith("data:image/jpeg;base64,")


def test_streaming_vision_request_rejects_unknown_arguments() -> None:
    with pytest.raises(ValidationError):
        VisionRequest(
            participant_id="alice",
            query="What is shown?",
            unsupported=True,
        )
