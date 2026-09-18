# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervisor routing eval: fake the five subagents and score the delegation.

Each case runs only the supervisor loop; subagents record the instruction they
receive and return a canned success. Scoring checks which agent was called and
that the instruction carries the resolved facts, so a routing regression
localises in seconds instead of a full nested rollout.

    uv run --project agent-samples/xr-render-demo/eval xr_render_demo_eval_supervisor [case ...]
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xr_ai_models import ChatMessage, ToolCall, load_models_config, make_llm
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import run_tool_loop
from xr_ai_voice import VoiceTurnController
from xr_render_demo_worker.agents.appearance.agent import DESCRIPTION as appearance_description
from xr_render_demo_worker.agents.memory.agent import DESCRIPTION as memory_description
from xr_render_demo_worker.agents.object.agent import DESCRIPTION as object_description
from xr_render_demo_worker.agents.placement.agent import DESCRIPTION as placement_description
from xr_render_demo_worker.agents.vision.agent import DESCRIPTION as vision_description
from xr_render_demo_worker.models import SceneRequest, SubagentResult, SubagentTask
from xr_render_demo_worker.supervisor import SceneSupervisor

from . import harness

_DESCRIPTIONS = {
    "placement_agent": placement_description,
    "appearance_agent": appearance_description,
    "object_agent": object_description,
    "vision_agent": vision_description,
    "memory_agent": memory_description,
}
_REFUSAL_CASE = "misrouted_subagent_refusal_recovers"
_SUPERVISOR_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "worker"
    / "xr_render_demo_worker"
    / "supervisor_prompt.txt"
).read_text(encoding="utf-8").strip()


def _make_fake_agent(name: str, description: str, calls: list) -> Tool:
    async def act(request: SubagentTask) -> SubagentResult:
        calls.append((name, request.instruction, request.reasoning_mode == "deliberate"))
        return SubagentResult(result="Done.")

    return Tool(name, description, SubagentTask, SubagentResult, act)


@dataclass(frozen=True)
class RoutingCase:
    name: str
    request: str
    scene: tuple[dict[str, Any], ...] = ()
    history: tuple[tuple[str, str], ...] = ()
    expect_agent: str = ""
    expect_agents: tuple[str, ...] = ()
    instruction_contains: tuple[str, ...] = ()
    instruction_forbids: tuple[str, ...] = ()
    forbid_agents: tuple[str, ...] = ()
    expect_supervisor_reasoning: bool | None = None
    expect_leaf_reasoning: bool | None = None


CASES = (
    RoutingCase(
        name="pronoun_resize_after_creation",
        request="Now double its size.",
        scene=(
            {"id": "sphere-1", "type": "sphere", "pos": [0.4, 1.5, -1.2], "color": [1, 1, 0], "size": 0.1},
            {"id": "box-0", "type": "box", "pos": [-0.6, 1.3, -1.6], "color": [0, 0.4, 1], "size": 0.15},
        ),
        history=(
            ("Make a yellow sphere.", "Added a yellow sphere (sphere-1)."),
        ),
        expect_agent="object_agent",
        instruction_contains=("sphere-1",),
    ),
    RoutingCase(
        name="pronoun_shrink_after_move",
        request="Make it half the size.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.2},
        ),
        history=(
            ("Make a blue cube.", "Added a blue cube (box-0)."),
            ("Move it left.", "Moved the cube (box-0) to your left."),
        ),
        expect_agent="object_agent",
        instruction_contains=("box-0",),
    ),
    RoutingCase(
        name="recolor_routes_to_appearance",
        request="Paint the sphere yellow.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
        ),
        expect_agent="appearance_agent",
    ),
    RoutingCase(
        name="move_routes_to_placement",
        request="Move the cube to the left.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
        ),
        expect_agent="placement_agent",
    ),
    RoutingCase(
        name="create_routes_to_object",
        request="Add a red sphere.",
        expect_agent="object_agent",
        expect_supervisor_reasoning=False,
        expect_leaf_reasoning=False,
    ),
    RoutingCase(
        name="remove_routes_to_object",
        request="Remove the blue cube.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
        ),
        expect_agent="object_agent",
    ),
    RoutingCase(
        name="vision_question_routes_to_vision",
        request="What color is the cup on my desk?",
        expect_agent="vision_agent",
    ),
    RoutingCase(
        name="physical_color_stays_with_mutating_agent",
        request="Turn the box the color of my carpet.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.5, -1.3], "color": [1, 1, 1], "size": 0.1},
        ),
        expect_agent="appearance_agent",
        instruction_contains=("carpet",),
    ),
    RoutingCase(
        name="memory_question_routes_to_memory",
        request="What did I ask you to make earlier?",
        history=(
            ("Add a red sphere.", "Added a red sphere."),
        ),
        expect_agent="memory_agent",
    ),
    RoutingCase(
        name="resize_routes_to_object_not_placement",
        request="Make it twice as big.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
    ),
    RoutingCase(
        name="swap_routes_to_placement",
        request="Swap the sphere and the cube.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "box-0", "type": "box", "pos": [0.5, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.1},
        ),
        expect_agent="placement_agent",
    ),
    RoutingCase(
        name="recolor_after_move_resolved_id",
        request="Make it green.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ),
        history=(
            ("Move the sphere left.", "Moved sphere-0 to your left."),
        ),
        expect_agent="appearance_agent",
        instruction_contains=("sphere-0",),
    ),
    RoutingCase(
        name="create_then_move_two_agents",
        request="Add a purple cube and then move it to my right.",
        expect_agent="object_agent",
    ),
    RoutingCase(
        name="nudge_routes_to_placement",
        request="Move the sphere forward a little.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ),
        expect_agent="placement_agent",
    ),
    RoutingCase(
        name="bare_create_after_work_stays_bare",
        request="Make a red cube.",
        history=(
            ("Add a cyan cube.", "Added a cyan cube."),
            ("Make a green sphere.", "Created a green sphere."),
        ),
        expect_agent="object_agent",
        instruction_forbids=("cyan cube", "green sphere"),
    ),
    RoutingCase(
        name="conversational_no_mutation",
        request="Can you help me add something to the scene?",
        forbid_agents=("object_agent", "appearance_agent", "placement_agent"),
    ),
    RoutingCase(
        name="row_creation_single_delegation",
        request="Make three red spheres in a row.",
        expect_agent="object_agent",
    ),
    RoutingCase(
        name="bare_create_after_work_no_extra_move",
        request="Make a red cube.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
    ),
    RoutingCase(
        name="correction_never_creates",
        request="That's the wrong sphere.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [-0.5, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-1", "type": "sphere", "pos": [0.5, 1.6, -1.5], "color": [0, 0, 1], "size": 0.1},
        ),
        history=(
            ("Add a red sphere above the blue sphere.", "Added a red box above the blue sphere."),
        ),
        forbid_agents=("object_agent", "appearance_agent", "placement_agent"),
    ),
    RoutingCase(
        name="put_new_object_is_creation_not_move",
        request="Put a blue square above the yellow square.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.6, -1.5], "color": [1, 1, 0], "size": 0.1},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
    ),
    RoutingCase(
        name="xr_object_color_never_uses_vision",
        request="Make the red box the same color as the green sphere.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "sphere-0", "type": "sphere", "pos": [0.5, 1.6, -1.5], "color": [0, 0.8, 0], "size": 0.1},
        ),
        expect_agent="appearance_agent",
        forbid_agents=("vision_agent",),
    ),
    RoutingCase(
        name="new_object_above_xr_object_no_vision",
        request="Put a blue sphere above the red capsule.",
        scene=(
            {"id": "capsule-0", "type": "capsule", "pos": [0.0, 1.5, -1.3], "color": [1, 0, 0], "size": 0.1},
        ),
        expect_agent="object_agent",
        forbid_agents=("vision_agent", "placement_agent"),
    ),
    RoutingCase(
        name="xr_object_position_never_uses_vision",
        request="Move the sphere to just above the red cube.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [0, 0, 1], "size": 0.1},
            {"id": "box-0", "type": "box", "pos": [0.5, 1.2, -1.5], "color": [1, 0, 0], "size": 0.1},
        ),
        expect_agent="placement_agent",
        forbid_agents=("vision_agent",),
    ),
    RoutingCase(
        name="physical_color_creation_stays_with_object_agent",
        request="Create a sphere that matches the color of my sleeve.",
        expect_agent="object_agent",
        instruction_contains=("sleeve",),
        forbid_agents=("vision_agent", "appearance_agent"),
    ),
    RoutingCase(
        name="past_physical_view_routes_to_vision",
        request="What was I holding ten seconds ago?",
        expect_agent="vision_agent",
        forbid_agents=("memory_agent",),
    ),
    RoutingCase(
        name="past_scene_fact_routes_to_memory",
        request="What color was the first object I created?",
        scene=(
            {
                "id": "sphere-0",
                "type": "sphere",
                "pos": [0.0, 1.6, -1.5],
                "color": [0, 0.4, 1],
                "size": 0.1,
            },
        ),
        history=(("Create a red sphere.", "Created sphere-0."),),
        expect_agent="memory_agent",
        forbid_agents=("vision_agent",),
    ),
    RoutingCase(
        name="current_scene_fact_is_answered_directly",
        request="What color is sphere-0 right now?",
        scene=(
            {
                "id": "sphere-0",
                "type": "sphere",
                "pos": [0.0, 1.6, -1.5],
                "color": [0, 0.4, 1],
                "size": 0.1,
            },
        ),
        forbid_agents=tuple(_DESCRIPTIONS),
    ),
    RoutingCase(
        name="acknowledgement_does_not_repeat_history",
        request="Okay.",
        history=(("Create a red sphere.", "Created sphere-0."),),
        forbid_agents=tuple(_DESCRIPTIONS),
    ),
    RoutingCase(
        name="compound_routes_each_domain_once",
        request="Move the ring left, recolor the cone orange, and create a blue sphere.",
        scene=(
            {
                "id": "ring-0",
                "type": "ring",
                "pos": [-0.5, 1.5, -1.5],
                "color": [1, 1, 1],
                "size": 0.1,
            },
            {
                "id": "cone-0",
                "type": "cone",
                "pos": [0.5, 1.5, -1.5],
                "color": [1, 1, 1],
                "size": 0.1,
            },
        ),
        expect_agents=("placement_agent", "appearance_agent", "object_agent"),
    ),
    # Held-out routing matrix. These paraphrases and boundary cases are kept
    # out of worked examples and candidate selection.
    RoutingCase(
        name="holdout_spawn_routes_to_object",
        request="Spawn a turquoise ring.",
        expect_agent="object_agent",
    ),
    RoutingCase(
        name="holdout_erase_routes_to_object",
        request="Erase the lavender capsule.",
        scene=(
            {"id": "capsule-4", "type": "capsule", "pos": [0.2, 1.4, -1.1], "color": [0.7, 0.5, 0.9], "size": 0.12},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
    ),
    RoutingCase(
        name="holdout_duplicate_routes_to_object",
        request="Duplicate the teal cone.",
        scene=(
            {"id": "cone-3", "type": "cone", "pos": [-0.4, 1.2, -1.8], "color": [0, 0.8, 0.8], "size": 0.1},
        ),
        expect_agent="object_agent",
    ),
    RoutingCase(
        name="holdout_reshape_routes_to_object",
        request="Turn the white ring into a capsule.",
        scene=(
            {"id": "ring-2", "type": "ring", "pos": [0.1, 1.3, -1.4], "color": [1, 1, 1], "size": 0.1},
        ),
        expect_agent="object_agent",
        forbid_agents=("appearance_agent", "placement_agent"),
        expect_leaf_reasoning=False,
    ),
    RoutingCase(
        name="holdout_existing_containment_routes_to_placement",
        request="Place the cyan ring inside the gray capsule.",
        scene=(
            {"id": "ring-5", "type": "ring", "pos": [-0.3, 1.4, -1.4], "color": [0, 1, 1], "size": 0.08},
            {"id": "capsule-6", "type": "capsule", "pos": [0.4, 1.4, -1.4], "color": [0.5, 0.5, 0.5], "size": 0.2},
        ),
        expect_agent="placement_agent",
        forbid_agents=("object_agent",),
    ),
    RoutingCase(
        name="holdout_new_containment_routes_to_object",
        request="Place a cyan ring inside the gray capsule.",
        scene=(
            {"id": "capsule-6", "type": "capsule", "pos": [0.4, 1.4, -1.4], "color": [0.5, 0.5, 0.5], "size": 0.2},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
    ),
    RoutingCase(
        name="holdout_physical_recolor_routes_to_appearance",
        request="Match the ring to the color of the mug beside me.",
        scene=(
            {"id": "ring-5", "type": "ring", "pos": [-0.3, 1.4, -1.4], "color": [1, 1, 1], "size": 0.08},
        ),
        expect_agent="appearance_agent",
        instruction_contains=("mug",),
        forbid_agents=("vision_agent",),
    ),
    RoutingCase(
        name="holdout_physical_create_routes_to_object",
        request="Build a cone matching what I am wearing.",
        expect_agent="object_agent",
        instruction_contains=("wearing",),
        forbid_agents=("vision_agent", "appearance_agent"),
    ),
    RoutingCase(
        name="holdout_live_physical_view_routes_to_vision",
        request="Is there an open doorway ahead of me right now?",
        expect_agent="vision_agent",
        forbid_agents=("memory_agent",),
    ),
    RoutingCase(
        name="holdout_past_physical_view_routes_to_vision",
        request="Was the doorway open twenty seconds ago?",
        expect_agent="vision_agent",
        forbid_agents=("memory_agent",),
    ),
    RoutingCase(
        name="holdout_original_scene_state_routes_to_memory",
        request="What shape was ring-5 before I changed it?",
        scene=(
            {"id": "ring-5", "type": "capsule", "pos": [-0.3, 1.4, -1.4], "color": [0, 1, 1], "size": 0.08},
        ),
        history=(("Change the ring into a capsule.", "Changed ring-5 into a capsule."),),
        expect_agent="memory_agent",
        forbid_agents=("vision_agent",),
    ),
    RoutingCase(
        name="holdout_current_scene_state_answered_directly",
        request="Where is ring-5 now?",
        scene=(
            {"id": "ring-5", "type": "ring", "pos": [-0.3, 1.4, -1.4], "color": [0, 1, 1], "size": 0.08},
        ),
        forbid_agents=tuple(_DESCRIPTIONS),
    ),
    RoutingCase(
        name="holdout_capability_question_does_not_mutate",
        request="Could you create objects if I asked you to?",
        forbid_agents=tuple(_DESCRIPTIONS),
    ),
    RoutingCase(
        name="holdout_negated_removal_does_not_mutate",
        request="Do not remove the capsule.",
        scene=(
            {"id": "capsule-6", "type": "capsule", "pos": [0.4, 1.4, -1.4], "color": [0.5, 0.5, 0.5], "size": 0.2},
        ),
        forbid_agents=("object_agent", "placement_agent", "appearance_agent"),
    ),
    RoutingCase(
        name="holdout_compound_object_and_appearance",
        request="Duplicate the teal cone, then make the ring orange.",
        scene=(
            {"id": "cone-3", "type": "cone", "pos": [-0.4, 1.2, -1.8], "color": [0, 0.8, 0.8], "size": 0.1},
            {"id": "ring-5", "type": "ring", "pos": [-0.3, 1.4, -1.4], "color": [1, 1, 1], "size": 0.08},
        ),
        expect_agents=("object_agent", "appearance_agent"),
    ),
    RoutingCase(
        name="holdout_compound_vision_and_placement",
        request="Tell me whether the doorway is open, then move the ring left.",
        scene=(
            {"id": "ring-5", "type": "ring", "pos": [-0.3, 1.4, -1.4], "color": [1, 1, 1], "size": 0.08},
        ),
        expect_agents=("vision_agent", "placement_agent"),
    ),
    RoutingCase(
        name="novel_arrangement_enables_leaf_reasoning",
        request=(
            "Rearrange the ring and cone into the most compact non-overlapping vertical "
            "composition, while keeping whichever is currently closer to me on top and "
            "preserving their left-to-right order as much as possible."
        ),
        scene=(
            {"id": "ring-5", "type": "ring", "pos": [-0.3, 1.4, -1.1], "color": [1, 1, 1], "size": 0.08},
            {"id": "cone-3", "type": "cone", "pos": [0.4, 1.2, -1.8], "color": [0, 0.8, 0.8], "size": 0.15},
        ),
        expect_agent="placement_agent",
        expect_supervisor_reasoning=False,
        expect_leaf_reasoning=True,
    ),
)


async def run_case(case: RoutingCase) -> bool:
    scene = harness.FakeScene.from_corpus_case(
        {"name": case.name, "scene": list(case.scene), "history": list(case.history), "user": case.request}
    )
    calls: list[tuple[str, str, bool]] = []
    fake_tools = [
        _make_fake_agent(name, desc, calls) for name, desc in _DESCRIPTIONS.items()
    ]
    llm = make_llm(load_models_config(harness.models_config_path()), "agent_llm")
    try:
        fake_scene, fake_tracking, fake_text_memory, _, _ = scene.make_tools()
        supervisor = SceneSupervisor(
            llm=llm,
            scene=fake_scene,
            tracking=fake_tracking,
            text_memory=fake_text_memory,
            subagent_tools=fake_tools,
        )
        errored = False
        controller = VoiceTurnController(
            turn_id=f"eval:{case.name}",
            timestamp_us=harness.EVAL_REFERENCE_US,
            publish=None,
            acknowledgement=True,
        )
        try:
            with controller.activate():
                reply = await supervisor.handle(
                    SceneRequest(
                        transcript=case.request,
                        participant_id="eval-user",
                        timestamp_us=harness.EVAL_REFERENCE_US,
                    )
                )
        except Exception as exc:
            reply = type("R", (), {"response": f"<workflow error: {exc}>"})()
            errored = True
    finally:
        await llm.close()

    called = [name for name, _instruction, _reasoning in calls]
    ok, why = True, "ok"
    if errored:
        ok, why = False, f"workflow error: {reply.response[:160]}"
    if case.expect_agent and case.expect_agent not in called:
        ok, why = False, f"{case.expect_agent} never called; called={called}"
    if case.expect_agents and tuple(called) != case.expect_agents:
        ok, why = False, f"expected agents={list(case.expect_agents)}, called={called}"
    for forbidden in case.forbid_agents:
        if forbidden in called:
            ok, why = False, f"{forbidden} called; called={called}"
    if (
        ok
        and case.expect_supervisor_reasoning is not None
        and controller.reasoning_enabled is not case.expect_supervisor_reasoning
    ):
        ok, why = False, (
            f"supervisor reasoning={controller.reasoning_enabled}, "
            f"expected {case.expect_supervisor_reasoning}"
        )
    if ok and case.expect_leaf_reasoning is not None:
        selected_reasoning = [
            reasoning for name, _instruction, reasoning in calls if name == case.expect_agent
        ]
        if selected_reasoning != [case.expect_leaf_reasoning]:
            ok, why = False, (
                f"leaf reasoning={selected_reasoning}, expected [{case.expect_leaf_reasoning}]"
            )
    if ok and case.instruction_forbids:
        instructions = " | ".join(
            i for name, i, _reasoning in calls if name == case.expect_agent
        )
        for needle in case.instruction_forbids:
            if needle.lower() in instructions.lower():
                ok, why = False, f"instruction contains forbidden {needle!r}: {instructions[:160]!r}"
    if ok and case.instruction_contains:
        instructions = " | ".join(
            i for name, i, _reasoning in calls if name == case.expect_agent
        )
        for needle in case.instruction_contains:
            if needle.lower() not in instructions.lower():
                ok, why = False, f"instruction missing {needle!r}: {instructions[:160]!r}"
    status = "PASS" if ok else f"FAIL {why}"
    detail = "; ".join(
        f"{name}(reasoning={reasoning}, {instruction[:80]})"
        for name, instruction, reasoning in calls
    )
    print(f"{status:32} {case.name}: {detail or reply.response}", flush=True)
    return ok


async def run_refusal_case() -> bool:
    """Verify that a structured leaf refusal causes a fresh delegation."""

    calls: list[tuple[str, str, bool]] = []
    tools = ToolSet(
        _make_fake_agent(name, description, calls)
        for name, description in _DESCRIPTIONS.items()
    )
    prior_call_id = "misrouted-placement"
    messages = (
        ChatMessage(role="system", content=_SUPERVISOR_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "Active participant: eval-user\n"
                f"Utterance timestamp: {harness.EVAL_REFERENCE_US}\n\n"
                "SCENE OBJECTS: none\n\n"
                "User request: Add a red sphere in front of me."
            ),
        ),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id=prior_call_id,
                    name="placement_agent",
                    arguments=(
                        '{"instruction":"Add a red sphere in front of me.",'
                        '"reasoning_mode":"fast"}'
                    ),
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content=SubagentResult(
                result="This is creation of a new object, not movement of an existing object.",
                handled=False,
                suggested_owner="object_agent",
            ).model_dump_json(),
            tool_call_id=prior_call_id,
        ),
    )
    llm = make_llm(load_models_config(harness.models_config_path()), "agent_llm")

    async def call_model(transcript, definitions):
        return await llm.chat(
            transcript,
            tools=list(definitions) or None,
            max_tokens=2048,
            temperature=0.0,
            enable_thinking=False,
        )

    try:
        try:
            result = await run_tool_loop(
                messages,
                tools,
                call_model,
                max_iterations=4,
            )
        except Exception as exc:
            print(f"FAIL workflow error: {exc!r:23} {_REFUSAL_CASE}", flush=True)
            return False
    finally:
        await llm.close()

    called = [name for name, _instruction, _reasoning in calls]
    ok = called == ["object_agent"]
    status = "PASS" if ok else f"FAIL called={called}"
    print(f"{status:32} {_REFUSAL_CASE}: {result.content}", flush=True)
    return ok


async def main() -> None:
    harness.audit_prompts()
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case names; omit to run all")
    args = parser.parse_args()
    wanted = set(args.cases)
    selected = [case for case in CASES if not wanted or case.name in wanted]
    run_refusal = not wanted or _REFUSAL_CASE in wanted
    known = {case.name for case in CASES} | {_REFUSAL_CASE}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"unknown cases: {args.cases}")
    results = [await run_case(case) for case in selected]
    if run_refusal:
        results.append(await run_refusal_case())
    print(f"\ndelegation: {sum(results)}/{len(results)} passed")
    if not all(results):
        raise SystemExit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
