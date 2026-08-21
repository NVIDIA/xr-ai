# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component eval: drive one focused subagent directly and score its mutations.

Each case invokes a single subagent Function with the supervisor-level
instruction it would receive (stable ids already resolved, facts from other
subagents already inlined), with no supervisor in the loop. Emitted
add/update/remove_primitive calls are matched order-independently against
expected argument ranges, so a regression localises to one agent and one
prompt instead of an end-to-end transcript.

    uv run --project agent-samples/xr-render-demo/eval xr_render_demo_eval_subagents
    uv run --project agent-samples/xr-render-demo/eval xr_render_demo_eval_subagents placement
    uv run --project agent-samples/xr-render-demo/eval xr_render_demo_eval_subagents swap_two_objects
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from xr_ai_models import load_models_config, make_llm
from xr_ai_tools.types import SpatialFrame, Vector3
from xr_render_demo_worker.agents import (
    make_appearance_agent,
    make_memory_agent,
    make_object_agent,
    make_placement_agent,
    make_vision_agent,
)
from xr_render_demo_worker.models import SubagentResult, SubagentTask
from xr_render_demo_worker.scene import SceneContext
from xr_render_scene import SceneObject

from . import harness

_PARTICIPANT = "eval-user"
_MUTATING = harness._MUTATING

# Fixture vocabulary stays distinct from prompt worked examples (see README).
_CONE = {
    "id": "cone-0",
    "type": "cone",
    "position": {"x": 0.5, "y": 1.4, "z": -1.6},
    "color": {"r": 1, "g": 1, "b": 1},
    "size": 0.1,
}
_RING = {
    "id": "ring-1",
    "type": "ring",
    "position": {"x": -1.0, "y": 1.4, "z": -1.6},
    "color": {"r": 1, "g": 1, "b": 1},
    "size": 0.1,
}
_CAPSULE = {
    "id": "capsule-2",
    "type": "capsule",
    "position": {"x": 2.0, "y": 1.4, "z": -1.6},
    "color": {"r": 1, "g": 1, "b": 1},
    "size": 0.1,
}


@dataclass(frozen=True)
class SubagentCase:
    name: str
    agent: str
    instruction: str
    scene: tuple[dict[str, Any], ...] = ()
    expect: tuple[dict[str, Any], ...] = ()
    recent_moves: tuple[str, ...] = ()
    pose: dict | None = None
    vision_answer: str = ""
    vision_error: str = ""
    memory: str = ""
    required_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    answer_contains: str = ""


# Expected args: a (lo, hi) tuple is an inclusive range, anything else is exact.
# The eval head pose is at (0, 1.6, 0) facing -z with +x to the user's right.
CASES = (
    SubagentCase(
        name="move_user_left",
        agent="placement",
        instruction="Move cone-0 one metre to my left.",
        scene=(_CONE,),
        # A stated distance shifts from the current position (x = 0.5 - 1.0).
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "cone-0", "x": (-0.65, -0.35)}},
        ),
    ),
    SubagentCase(
        name="move_next_to_object",
        agent="placement",
        instruction="Move ring-1 next to cone-0.",
        scene=(_CONE, _RING),
        expect=(
            {
                "tool": "update_primitive",
                "args": {"obj_id": "ring-1", "x": (0.0, 1.0), "z": (-2.1, -1.1)},
            },
        ),
    ),
    SubagentCase(
        name="move_between_objects",
        agent="placement",
        instruction="Move ring-1 halfway between cone-0 and capsule-2.",
        scene=(_CONE, _RING, _CAPSULE),
        expect=(
            {
                "tool": "update_primitive",
                "args": {"obj_id": "ring-1", "x": (1.15, 1.35), "z": (-1.7, -1.5)},
            },
        ),
    ),
    SubagentCase(
        name="stack_on_object",
        agent="placement",
        instruction="Put ring-1 on cone-0.",
        scene=(_CONE, _RING),
        # Stacking, not containment: the ring must end above the cone.
        expect=(
            {
                "tool": "update_primitive",
                "args": {"obj_id": "ring-1", "x": (0.4, 0.6), "y": (1.45, 2.0), "z": (-1.7, -1.5)},
            },
        ),
    ),
    SubagentCase(
        name="move_into_container",
        agent="placement",
        instruction="Put ring-1 in capsule-2.",
        scene=(_CONE, _RING, _CAPSULE),
        expect=(
            {
                "tool": "update_primitive",
                "args": {"obj_id": "ring-1", "x": (1.95, 2.05), "y": (1.35, 1.45), "z": (-1.65, -1.55)},
            },
        ),
    ),
    SubagentCase(
        name="move_toward_object",
        agent="placement",
        instruction="Move ring-1 closer to cone-0.",
        scene=(_CONE, _RING),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "ring-1", "x": (-0.95, 0.45)}},
        ),
    ),
    SubagentCase(
        name="put_object_back",
        agent="placement",
        instruction="Put cone-0 back where it was before.",
        scene=(dict(_CONE, position={"x": 0.5, "y": 1.4, "z": -0.5}),),
        recent_moves=("cone-0: previously at (0.5, 1.4, -1.6), now at (0.5, 1.4, -0.5)",),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "cone-0", "z": (-1.7, -1.5)}},
        ),
    ),
    SubagentCase(
        name="unresolvable_referent_reports_back",
        agent="placement",
        instruction="Place the blue sphere above the green sphere.",
        scene=(
            dict(_CONE, id="sphere-0", type="sphere", color={"r": 1, "g": 1, "b": 1}),
            dict(_RING, id="sphere-1", type="sphere", color={"r": 0, "g": 0.8, "b": 0}),
        ),
        forbid_tools=("update_primitive", "add_primitive", "remove_primitive"),
        answer_contains="blue",
    ),
    SubagentCase(
        name="move_yellow_not_red",
        agent="placement",
        instruction="Move the yellow cube down one meter.",
        scene=(
            dict(_CONE, id="box-0", type="box", color={"r": 1, "g": 0, "b": 0},
                 position={"x": 0.1, "y": 1.3, "z": -1.0}),
            dict(_RING, id="box-1", type="box", color={"r": 1, "g": 1, "b": 0},
                 position={"x": -0.9, "y": 2.8, "z": -0.3}),
        ),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "box-1", "y": (1.7, 1.9)}},
        ),
    ),
    SubagentCase(
        name="swap_two_objects",
        agent="placement",
        instruction="Swap the positions of cone-0 and ring-1.",
        scene=(_CONE, _RING),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "cone-0", "x": (-1.05, -0.95)}},
            {"tool": "update_primitive", "args": {"obj_id": "ring-1", "x": (0.45, 0.55)}},
        ),
    ),
    SubagentCase(
        name="create_at_position",
        agent="object",
        instruction="Create a white box at x=0, y=1.5, z=-1.2 with size 0.1.",
        expect=(
            {
                "tool": "add_primitive",
                "args": {
                    "prim_type": "box",
                    "x": (-0.05, 0.05),
                    "y": (1.45, 1.55),
                    "z": (-1.25, -1.15),
                },
            },
        ),
    ),
    SubagentCase(
        name="double_named_object",
        agent="object",
        instruction="Double the size of ring-1.",
        scene=(_CONE, _RING),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "ring-1", "size": (0.19, 0.21)}},
        ),
    ),
    SubagentCase(
        name="shrink_named_object",
        agent="object",
        instruction="Make the box-0 object half its current size.",
        scene=(
            {
                "id": "box-0",
                "type": "box",
                "position": {"x": 0.0, "y": 1.6, "z": -1.5},
                "color": {"r": 0, "g": 0.4, "b": 1},
                "size": 0.2,
            },
        ),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "box-0", "size": (0.09, 0.11)}},
        ),
    ),
    SubagentCase(
        name="row_of_three",
        agent="object",
        instruction="Create three red spheres in a row at eye height, 0.5 metres apart, in front of the user.",
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "sphere", "r": (0.7, 1.0), "g": (0.0, 0.4)}},
            {"tool": "add_primitive", "args": {"prim_type": "sphere", "r": (0.7, 1.0), "g": (0.0, 0.4)}},
            {"tool": "add_primitive", "args": {"prim_type": "sphere", "r": (0.7, 1.0), "g": (0.0, 0.4)}},
        ),
    ),
    SubagentCase(
        name="raise_above_anchor_30cm",
        agent="placement",
        instruction="Move ring-1 thirty centimetres above cone-0.",
        scene=(_CONE, _RING),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "ring-1", "y": (1.65, 1.75)}},
        ),
    ),
    SubagentCase(
        name="create_stated_distance_left",
        agent="object",
        instruction="Create a white sphere one metre to the user's left at eye height.",
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "sphere", "x": (-1.05, -0.95)}},
        ),
    ),
    SubagentCase(
        name="create_at_feet",
        agent="object",
        instruction="Create a red sphere on the floor at the user's feet.",
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "sphere", "r": (0.7, 1.0), "y": (-0.05, 0.5)}},
        ),
    ),
    SubagentCase(
        name="create_bare",
        agent="object",
        instruction="Create a blue sphere with no position stated.",
        expect=(
            {
                "tool": "add_primitive",
                "args": {"prim_type": "sphere", "x": (-0.05, 0.05), "y": (1.55, 1.65), "z": (-1.55, -1.45)},
            },
        ),
    ),
    SubagentCase(
        name="create_cube_means_box",
        agent="object",
        instruction="Make a green cube.",
        expect=(
            {
                "tool": "add_primitive",
                "args": {"prim_type": "box", "g": (0.5, 1.0), "r": (0.0, 0.4), "b": (0.0, 0.4), "y": (1.4, 1.8)},
            },
        ),
    ),
    SubagentCase(
        name="create_bare_with_objects",
        agent="object",
        instruction="Create a green cube with no position stated.",
        scene=(_CONE, _RING),
        expect=(
            {
                "tool": "add_primitive",
                "args": {"prim_type": "box", "x": (-0.05, 0.05), "y": (1.55, 1.65), "z": (-1.55, -1.45)},
            },
        ),
    ),
    SubagentCase(
        name="create_bare_crowded_front_occupied",
        agent="object",
        instruction="Add a green cube",
        scene=(
            dict(_CONE, position={"x": 0.0, "y": 1.6, "z": -1.5}),
            dict(_RING, id="ring-1", position={"x": 0.0, "y": 1.9, "z": -1.5}),
            dict(_CAPSULE, id="capsule-2", position={"x": 0.0, "y": 2.2, "z": -1.5}),
        ),
        expect=(
            {
                "tool": "add_primitive",
                "args": {"prim_type": "box", "x": (-0.05, 0.05), "y": (1.55, 1.65), "z": (-1.55, -1.45)},
            },
        ),
    ),
    SubagentCase(
        name="create_cube_off_origin_pose",
        agent="object",
        instruction="Create a green cube with no position stated.",
        pose={"position": {"x": 2.0, "y": 1.6, "z": 1.5}},
        expect=(
            {
                "tool": "add_primitive",
                "args": {"prim_type": "box", "x": (1.95, 2.05), "y": (1.55, 1.65), "z": (-0.05, 0.05)},
            },
        ),
    ),
    SubagentCase(
        name="anchored_create_right_color",
        agent="object",
        instruction="Create a red box above the blue sphere.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "position": {"x": -1.0, "y": 1.6, "z": -1.5},
             "color": {"r": 0, "g": 0, "b": 1}, "size": 0.1},
            {"id": "sphere-1", "type": "sphere", "position": {"x": 1.0, "y": 1.6, "z": -1.5},
             "color": {"r": 0, "g": 0.8, "b": 0}, "size": 0.1},
        ),
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "box", "x": (-1.1, -0.9), "y": (1.65, 2.0)}},
        ),
    ),
    SubagentCase(
        name="create_garbled_shape_noun",
        agent="object",
        # STT corruption of the created shape itself: "sphere" heard as
        # "spear" with no anchor stated. Must stay a bare user-front create.
        instruction="Create one blue spear",
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "sphere", "r": (0, 0.05), "b": (0.9, 1.0),
                                               "x": (-0.2, 0.2), "z": (-1.7, -1.3)}},
        ),
    ),
    SubagentCase(
        name="anchored_create_garbled_anchor_noun",
        agent="object",
        # STT corruption: "sphere" heard as "spear". Nearest scene match wins.
        instruction="Create a red cube above the green spear.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "position": {"x": -1.0, "y": 1.6, "z": -1.5},
             "color": {"r": 0, "g": 0.8, "b": 0}, "size": 0.1},
            {"id": "sphere-1", "type": "sphere", "position": {"x": 1.0, "y": 1.6, "z": -1.5},
             "color": {"r": 0, "g": 0, "b": 1}, "size": 0.1},
        ),
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "box", "x": (-1.1, -0.9), "y": (1.65, 2.0)}},
        ),
    ),
    SubagentCase(
        name="remove_object",
        agent="object",
        instruction="Remove ring-1.",
        scene=(_CONE, _RING),
        expect=({"tool": "remove_primitive", "args": {"obj_id": "ring-1"}},),
    ),
    SubagentCase(
        name="resize_object",
        agent="object",
        instruction="Make cone-0 twice as big.",
        scene=(_CONE,),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "cone-0", "size": (0.19, 0.21)}},
        ),
    ),
    SubagentCase(
        name="duplicate_object",
        agent="object",
        instruction="Create an identical copy of cone-0 beside it.",
        scene=(_CONE,),
        expect=(
            {"tool": "add_primitive", "args": {"prim_type": "cone", "size": (0.05, 0.15)}},
        ),
    ),
    SubagentCase(
        name="change_shape",
        agent="object",
        instruction="Change cone-0 into a box.",
        scene=(_CONE,),
        expect=(
            {"tool": "update_primitive", "args": {"obj_id": "cone-0", "prim_type": "box"}},
        ),
    ),
    SubagentCase(
        name="recolor_explicit_rgb",
        agent="appearance",
        instruction="Set cone-0 to the observed wall color: normalized RGB (1.0, 0.5, 0.0).",
        scene=(_CONE,),
        expect=(
            {
                "tool": "update_primitive",
                "args": {
                    "obj_id": "cone-0",
                    "r": (0.95, 1.0),
                    "g": (0.45, 0.55),
                    "b": (0.0, 0.05),
                },
            },
        ),
    ),
    SubagentCase(
        name="recolor_by_name",
        agent="appearance",
        instruction="Make ring-1 orange.",
        scene=(_CONE, _RING),
        expect=(
            {
                "tool": "update_primitive",
                "args": {"obj_id": "ring-1", "r": (0.8, 1.0), "g": (0.3, 0.7), "b": (0.0, 0.3)},
            },
        ),
    ),
    SubagentCase(
        name="live_color_question",
        agent="vision",
        instruction="What color is the object the user is holding right now?",
        vision_answer="The user is holding a bright red apple.",
        required_tools=("look_at_current_frame",),
        answer_contains="red",
    ),
    SubagentCase(
        name="past_color_question",
        agent="vision",
        instruction="What color was the object the user held ten seconds before the utterance timestamp?",
        vision_answer="The previously held object was purple.",
        required_tools=("look_at_past_frame",),
        answer_contains="purple",
    ),
    SubagentCase(
        name="vision_dead_camera_degrades",
        agent="vision",
        instruction=(
            "What physical objects or surfaces are directly in front of the user for placing a sphere "
            "two meters ahead?"
        ),
        vision_error="No camera frame available.",
        required_tools=("look_at_current_frame",),
        answer_contains="no camera",
    ),
    SubagentCase(
        name="vision_answers_scene_from_data",
        agent="vision",
        instruction="Is there a red box currently present in the scene, and if so, what is its position?",
        scene=(
            {
                "id": "sphere-0",
                "type": "sphere",
                "position": {"x": 0.0, "y": 1.6, "z": -1.5},
                "color": {"r": 0, "g": 0, "b": 1},
                "size": 0.1,
            },
        ),
        answer_contains="no red box",
    ),
    SubagentCase(
        name="recall_earlier_session",
        agent="memory",
        instruction="What object did we discuss in the earlier session?",
        memory="We discussed a small cyan sphere.",
        required_tools=("recall_conversation",),
        answer_contains="cyan",
    ),
    SubagentCase(
        name="recolor_multiple",
        agent="appearance",
        instruction="Make cone-0 and ring-1 magenta.",
        scene=(_CONE, _RING),
        expect=(
            {
                "tool": "update_primitive",
                "args": {"obj_id": "cone-0", "r": (0.8, 1.0), "g": (0.0, 0.3), "b": (0.8, 1.0)},
            },
            {
                "tool": "update_primitive",
                "args": {"obj_id": "ring-1", "r": (0.8, 1.0), "g": (0.0, 0.3), "b": (0.8, 1.0)},
            },
        ),
    ),
)

def _make_agent(
    case_agent, llm, fake_scene, fake_tracking, fake_text_memory,
    fake_current_frame, fake_image_query, context,
):
    if case_agent == "placement":
        return make_placement_agent(llm, fake_scene, fake_tracking, context)
    if case_agent == "appearance":
        return make_appearance_agent(llm, fake_scene, context)
    if case_agent == "object":
        return make_object_agent(llm, fake_scene, fake_tracking, context)
    if case_agent == "vision":
        return make_vision_agent(llm, fake_current_frame, fake_image_query, context)
    if case_agent == "memory":
        return make_memory_agent(llm, fake_text_memory)
    raise ValueError(f"unknown agent: {case_agent!r}")


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, want in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if isinstance(want, tuple):
            low, high = want
            if not (isinstance(got, (int, float)) and low <= got <= high):
                return False
        elif got != want:
            return False
    return True


def check(calls: list[tuple[str, dict[str, Any]]], case: SubagentCase, reply: str) -> tuple[bool, str]:
    """Match expected mutations order-independently; reject any extra mutation."""
    names = {name for name, _args in calls}
    missing = set(case.required_tools) - names
    if missing:
        return False, f"missing tools: {sorted(missing)}"
    if hit := set(case.forbid_tools) & names:
        return False, f"forbidden tools called: {sorted(hit)}"
    if case.answer_contains and case.answer_contains.lower() not in reply.lower():
        return False, f"reply does not mention {case.answer_contains!r}"
    mutations = [(name, args) for name, args in calls if name in _MUTATING]
    expected_adds = sum(1 for item in case.expect if item["tool"] == "add_primitive")
    actual_adds = sum(1 for name, _args in mutations if name == "add_primitive")
    if actual_adds > expected_adds:
        actual = [f"{name}({args})" for name, args in mutations]
        return False, f"duplicate add: {actual_adds} adds for {expected_adds} expected | actual: {actual}"
    remaining = list(mutations)
    for item in case.expect:
        for index, (name, args) in enumerate(remaining):
            if name == item["tool"] and _args_match(args, item["args"]):
                remaining.pop(index)
                break
        else:
            actual = [f"{name}({args})" for name, args in mutations]
            return False, f"unmatched {item['tool']}({item['args']}) | actual: {actual}"
    if remaining:
        actual = [f"{name}({args})" for name, args in remaining]
        return False, f"unexpected mutation(s): {actual}"
    return True, f"matched {len(case.expect)} mutation(s)"


def _default_pose(override: dict | None = None) -> SpatialFrame:
    base = harness._DEFAULT_POSE
    merged = {**base, **(override or {})}
    return SpatialFrame(
        origin=Vector3(**merged["position"]),
        forward=Vector3(**merged["forward"]),
        right=Vector3(**merged["right"]),
        up=Vector3(**merged["up"]),
    )


async def run_case(case: SubagentCase) -> bool:
    objects = [SceneObject.model_validate(item) for item in case.scene]
    scene = harness.FakeScene(
        {item.id: item for item in objects},
        _default_pose(case.pose),
        case.vision_answer,
        case.vision_error,
        case.memory,
    )
    llm = make_llm(load_models_config(harness._CONFIG.models_config), "agent_llm")
    try:
        fake_scene, fake_tracking, fake_text_memory, fake_current_frame, fake_image_query = scene.make_tools()
        context = SceneContext(fake_scene, fake_tracking)
        context._recent_moves[_PARTICIPANT] = list(case.recent_moves)
        agent = _make_agent(
            case.agent, llm, fake_scene, fake_tracking, fake_text_memory,
            fake_current_frame, fake_image_query, context,
        )
        errored = False
        try:
            reply = await agent.execute(
                SubagentTask(
                    instruction=case.instruction,
                    participant_id=_PARTICIPANT,
                    reference_time_us=10_000_000,
                )
            )
        except Exception as exc:
            reply = SubagentResult(result=f"<workflow error: {exc}>")
            errored = True
    finally:
        await llm.close()
    ok, why = check(scene.calls, case, reply.result)
    if errored:
        ok, why = False, f"workflow error | {why}"
    status = "PASS" if ok else f"FAIL {why}"
    print(f"{status:32} {case.agent}/{case.name}: {reply.result}")
    return ok


async def main() -> None:
    harness.audit_prompts()
    parser = argparse.ArgumentParser()
    parser.add_argument("filter", nargs="?", help="Agent or case name; omit to run all cases")
    args = parser.parse_args()
    selected = [case for case in CASES if args.filter in (None, case.agent, case.name)]
    if not selected:
        raise SystemExit(f"unknown agent or case: {args.filter}")
    passed = [await run_case(case) for case in selected]
    print(f"\nsubagents: {sum(passed)}/{len(passed)} passed")
    if not all(passed):
        raise SystemExit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
