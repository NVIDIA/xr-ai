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
from typing import Any

from xr_ai_models import load_models_config, make_llm
from xr_ai_tools import Tool
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


def _make_fake_agent(name: str, description: str, calls: list) -> Tool:
    async def act(request: SubagentTask) -> SubagentResult:
        calls.append((name, request.instruction))
        return SubagentResult(result="Done.")

    return Tool(name, description, SubagentTask, SubagentResult, act)


@dataclass(frozen=True)
class RoutingCase:
    name: str
    request: str
    scene: tuple[dict[str, Any], ...] = ()
    history: tuple[tuple[str, str], ...] = ()
    expect_agent: str = ""
    instruction_contains: tuple[str, ...] = ()
    instruction_forbids: tuple[str, ...] = ()
    forbid_agents: tuple[str, ...] = ()


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
)


async def run_case(case: RoutingCase) -> bool:
    scene = harness.FakeScene.from_corpus_case(
        {"name": case.name, "scene": list(case.scene), "history": list(case.history), "user": case.request}
    )
    calls: list[tuple[str, str]] = []
    fake_tools = [
        _make_fake_agent(name, desc, calls) for name, desc in _DESCRIPTIONS.items()
    ]
    llm = make_llm(load_models_config(harness._CONFIG.models_yaml), "agent_llm")
    try:
        fake_scene, fake_tracking, fake_text_memory, _, _ = scene.make_tools()
        supervisor = SceneSupervisor(
            llm=llm,
            scene=fake_scene,
            tracking=fake_tracking,
            text_memory=fake_text_memory,
            subagent_tools=fake_tools,
        )
        try:
            reply = await supervisor.handle(
                SceneRequest(
                    transcript=case.request,
                    participant_id="eval-user",
                    timestamp_us=10_000_000,
                )
            )
        except Exception as exc:
            reply = type("R", (), {"response": f"<workflow error: {exc}>"})()
    finally:
        await llm.close()

    called = [name for name, _instruction in calls]
    ok, why = True, "ok"
    if case.expect_agent and case.expect_agent not in called:
        ok, why = False, f"{case.expect_agent} never called; called={called}"
    for forbidden in case.forbid_agents:
        if forbidden in called:
            ok, why = False, f"{forbidden} called; called={called}"
    if ok and case.instruction_forbids:
        instructions = " | ".join(i for name, i in calls if name == case.expect_agent)
        for needle in case.instruction_forbids:
            if needle.lower() in instructions.lower():
                ok, why = False, f"instruction contains forbidden {needle!r}: {instructions[:160]!r}"
    if ok and case.instruction_contains:
        instructions = " | ".join(i for name, i in calls if name == case.expect_agent)
        for needle in case.instruction_contains:
            if needle.lower() not in instructions.lower():
                ok, why = False, f"instruction missing {needle!r}: {instructions[:160]!r}"
    status = "PASS" if ok else f"FAIL {why}"
    detail = "; ".join(f"{name}({instruction[:80]})" for name, instruction in calls)
    print(f"{status:32} {case.name}: {detail or reply.response}", flush=True)
    return ok


async def main() -> None:
    harness.audit_prompts()
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case names; omit to run all")
    args = parser.parse_args()
    wanted = set(args.cases)
    selected = [case for case in CASES if not wanted or case.name in wanted]
    if not selected:
        raise SystemExit(f"unknown cases: {args.cases}")
    results = [await run_case(case) for case in selected]
    print(f"\ndelegation: {sum(results)}/{len(results)} passed")
    if not all(results):
        raise SystemExit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
