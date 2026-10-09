# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the eval harness's scoring primitives; a silent checker bug voids every tier score."""

from xr_render_demo_eval.subagents import _args_match


def test_exact_values_match():
    assert _args_match({"obj_id": "cone-7", "x": 1.0}, {"obj_id": "cone-7"})
    assert not _args_match({"obj_id": "cone-7"}, {"obj_id": "ring-1"})


def test_range_is_inclusive_at_both_ends():
    assert _args_match({"x": 0.5}, {"x": (0.5, 1.0)})
    assert _args_match({"x": 1.0}, {"x": (0.5, 1.0)})
    assert not _args_match({"x": 1.01}, {"x": (0.5, 1.0)})


def test_missing_key_fails():
    assert not _args_match({"x": 0.5}, {"y": (0.0, 1.0)})


def test_offline_vision_eval_uses_current_model_facing_tools():
    import asyncio

    from xr_ai_models import ChatResponse
    from xr_render_demo_worker.agents.vision.agent import _LIVE_ONLY_DESCRIPTION, make_vision_agent
    from xr_render_demo_worker.models import SubagentTask
    from xr_render_demo_eval import harness
    from xr_render_demo_eval.supervisor import _DESCRIPTIONS

    class CapturingLLM:
        def __init__(self):
            self.tools = ()

        async def chat(self, messages, *, tools=None, **kwargs):
            self.tools = tuple(tools or ())
            return ChatResponse(
                content="Recorded video is not available.",
                reasoning=None,
                tool_calls=None,
                finish_reason="stop",
                raw={"model": "stub"},
            )

    async def run_case():
        llm = CapturingLLM()
        scene = harness.FakeScene.from_case(
            harness.Case(name="vision-tool-contract", request="",)
        )
        fake_scene, fake_tracking, fake_memory, current_frame, image_query = scene.make_tools()
        del fake_scene, fake_tracking, fake_memory
        vision = make_vision_agent(llm, current_frame, image_query, video=None)
        assert vision.description == _LIVE_ONLY_DESCRIPTION
        reply = await vision.handler(SubagentTask(instruction="Describe the current view."))
        assert reply.result == "Recorded video is not available."
        return llm.tools

    definitions = asyncio.run(run_case())
    names = {definition.name for definition in definitions}
    assert names == {"look_at_current_frame"}
    assert _DESCRIPTIONS["vision_agent"] == _LIVE_ONLY_DESCRIPTION


def test_disabled_recorded_video_scenario_is_reported_deferred(monkeypatch, capsys):
    import asyncio
    import sys

    from xr_render_demo_eval import harness

    assert not harness._CONFIG.video_history_enabled
    monkeypatch.setattr(sys, "argv", ["eval", "historical_vision"])
    asyncio.run(harness.main())

    output = capsys.readouterr().out
    assert "DEFERRED historical_vision" in output
    assert "deferred: 1" in output
    assert "scenarios:" not in output
