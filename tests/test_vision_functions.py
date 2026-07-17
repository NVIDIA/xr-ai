# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the native image-question vision function."""

from __future__ import annotations

import base64
from types import SimpleNamespace

from nat.builder.workflow_builder import WorkflowBuilder
from PIL import Image
from xr_ai_nat.functions.vision import VisionFunctionsConfig


class _Vlm:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[str, str, str]] = []

    async def ask_image(self, image: str, question: str, *, system_prompt: str = ""):
        self.calls.append((image, question, system_prompt))
        return SimpleNamespace(content=self.content)


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
