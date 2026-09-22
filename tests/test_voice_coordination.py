# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for model-selected turn coordination."""

from __future__ import annotations

from xr_ai_models import ChatResponse
from xr_ai_tools import ToolSet
from xr_ai_voice import VoiceOutput, VoiceTurnController
from xr_ai_voice._coordination import _acknowledge_if_needed


async def test_acknowledge_speaks_model_text_only_once() -> None:
    outputs: list[VoiceOutput] = []

    async def publish(output: VoiceOutput) -> None:
        outputs.append(output)

    controller = VoiceTurnController(
        turn_id="turn-1",
        timestamp_us=42,
        publish=publish,
    )
    assert await controller.acknowledge("I’ll reconcile those changes.") is True
    assert await controller.acknowledge("I’ll say this twice.") is False
    assert outputs == [
        VoiceOutput(
            text="I’ll reconcile those changes.",
            timestamp_us=42,
            kind="acknowledgement",
            turn_id="turn-1",
        )
    ]


async def test_progress_tool_speaks_model_text() -> None:
    outputs: list[VoiceOutput] = []

    async def publish(output: VoiceOutput) -> None:
        outputs.append(output)

    controller = VoiceTurnController(
        turn_id="turn-1",
        timestamp_us=None,
        publish=publish,
    )
    progress = controller.extend(ToolSet(())).get("turn__report_progress")
    assert progress is not None
    result = await progress.invoke('{"message":"The first change is complete."}')

    assert '"accepted":true' in result.content
    assert outputs == [
        VoiceOutput(
            text="The first change is complete.",
            timestamp_us=None,
            kind="progress",
            turn_id="turn-1",
        )
    ]


async def test_model_acknowledgement_can_speak_or_stay_silent() -> None:
    outputs: list[VoiceOutput] = []

    async def publish(output: VoiceOutput) -> None:
        outputs.append(output)

    class Llm:
        def __init__(self) -> None:
            self.responses = [
                "SILENT",
                "I'll check the current scene.",
            ]
            self.calls: list[dict] = []
            self.messages = None

        async def chat(self, messages, **kwargs):
            self.calls.append(kwargs)
            self.messages = messages
            return ChatResponse(
                content=self.responses.pop(0),
                reasoning=None,
                tool_calls=None,
                finish_reason="stop",
                raw={},
            )

    controller = VoiceTurnController(
        turn_id="turn-1",
        timestamp_us=42,
        publish=publish,
    )
    llm = Llm()

    await _acknowledge_if_needed(controller, llm, "Hello", context="scene")
    await _acknowledge_if_needed(controller, llm, "Inspect this", context="scene")

    assert outputs == [
        VoiceOutput(
            text="I'll check the current scene.",
            timestamp_us=42,
            kind="acknowledgement",
            turn_id="turn-1",
        )
    ]
    assert all(call["enable_thinking"] is False for call in llm.calls)
    assert all(call["max_tokens"] == 48 for call in llm.calls)
    assert all("tools" not in call for call in llm.calls)
    assert "starting a watch" not in llm.messages[0].content
    assert llm.messages[1].content == "Area: scene\nCurrent user request: Inspect this"
