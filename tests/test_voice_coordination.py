# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for model-selected turn coordination."""

from __future__ import annotations

from xr_ai_models import ChatMessage
from xr_ai_tools import ToolSet
from xr_ai_voice import VoiceOutput, VoiceTurnController


async def test_prepare_work_enables_reasoning_and_speaks_model_text() -> None:
    outputs: list[VoiceOutput] = []

    async def publish(output: VoiceOutput) -> None:
        outputs.append(output)

    controller = VoiceTurnController(
        turn_id="turn-1",
        timestamp_us=42,
        publish=publish,
        acknowledgement=True,
    )
    tools = controller.extend(ToolSet(()))
    prepare = tools.get("turn__prepare_work")
    assert prepare is not None

    result = await prepare.invoke(
        '{"message":"I’ll reconcile those changes.","use_reasoning":true}'
    )

    assert '"accepted":true' in result.content
    assert controller.reasoning_enabled is True
    assert outputs == [
        VoiceOutput(
            text="I’ll reconcile those changes.",
            timestamp_us=42,
            kind="acknowledgement",
            turn_id="turn-1",
        )
    ]


async def test_reasoning_guidance_appears_only_after_model_selects_it() -> None:
    controller = VoiceTurnController(
        turn_id="turn-1",
        timestamp_us=None,
        publish=None,
        acknowledgement=False,
    )
    messages = (ChatMessage(role="system", content="Domain contract."),)
    assert controller.messages(messages) == messages

    prepare = controller.extend(ToolSet(())).get("turn__prepare_work")
    assert prepare is not None
    await prepare.invoke('{"message":"I’m resolving the constraints.","use_reasoning":true}')

    guided = controller.messages(messages)
    assert "Domain contract." in str(guided[0].content)
    assert "Reason only about unresolved choices" in str(guided[0].content)


async def test_prepare_work_is_accepted_only_once() -> None:
    outputs: list[VoiceOutput] = []

    async def publish(output: VoiceOutput) -> None:
        outputs.append(output)

    controller = VoiceTurnController(
        turn_id="turn-1",
        timestamp_us=None,
        publish=publish,
        acknowledgement=True,
    )
    prepare = controller.extend(ToolSet(())).get("turn__prepare_work")
    assert prepare is not None

    first = await prepare.invoke('{"message":"I’ll check.","use_reasoning":false}')
    second = await prepare.invoke('{"message":"Checking again.","use_reasoning":true}')

    assert '"accepted":true' in first.content
    assert '"accepted":false' in second.content
    assert controller.reasoning_enabled is False
    assert [output.text for output in outputs] == ["I’ll check."]
