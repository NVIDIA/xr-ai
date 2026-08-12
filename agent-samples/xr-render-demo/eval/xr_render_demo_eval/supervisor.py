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

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, FunctionRef, LLMRef, register_function
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from pydantic import ConfigDict, Field
from xr_ai_models import load_models_config, make_llm
from xr_ai_nat.functions.text_memory import RecallConversationRequest
from xr_ai_nat.llm import ModelsLLMConfig
from xr_render_demo_worker.agents.appearance.agent import DESCRIPTION as appearance_description
from xr_render_demo_worker.agents.memory.agent import DESCRIPTION as memory_description
from xr_render_demo_worker.agents.object.agent import DESCRIPTION as object_description
from xr_render_demo_worker.agents.placement.agent import DESCRIPTION as placement_description
from xr_render_demo_worker.agents.vision.agent import DESCRIPTION as vision_description
from xr_render_demo_worker.models import SceneReply, SubagentResult, SubagentTask

from . import harness

_DESCRIPTIONS = {
    "placement_agent": placement_description,
    "appearance_agent": appearance_description,
    "object_agent": object_description,
    "vision_agent": vision_description,
    "memory_agent": memory_description,
}
_PROMPT = (
    Path(__file__).resolve().parent
    / "../../worker/xr_render_demo_worker/supervisor_prompt.txt"
).resolve()


class _FakeAgentConfig(FunctionBaseConfig, name="xr_render_eval_fake_agent"):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    agent_name: str
    description: str
    recorder: Any = Field(exclude=True, repr=False)


@register_function(config_type=_FakeAgentConfig)
async def _fake_agent(config: _FakeAgentConfig, _builder: Builder):
    async def act(request: SubagentTask) -> SubagentResult:
        config.recorder.append((config.agent_name, request.instruction))
        return SubagentResult(result="Done.")

    yield FunctionInfo.from_fn(act, description=config.description)


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
        name="bare_pronoun_double_size",
        request="Double its size.",
        scene=(
            {"id": "sphere-1", "type": "sphere", "pos": [0.13, 1.8, -1.59], "color": [0, 0, 1], "size": 0.1},
        ),
        expect_agent="object_agent",
        instruction_contains=("sphere-1",),
    ),
    RoutingCase(
        name="bare_pronoun_half_size",
        request="Make it half the size.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [0.0, 1.6, -1.5], "color": [0, 0.4, 1], "size": 0.2},
        ),
        expect_agent="object_agent",
        instruction_contains=("box-0",),
    ),
    RoutingCase(
        name="move_existing_routes_to_placement",
        request="Put the sphere in the cube.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [1.0, 1.6, -1.5], "color": [1, 0, 0], "size": 0.1},
            {"id": "box-0", "type": "box", "pos": [-0.5, 1.3, -1.5], "color": [0, 0.4, 1], "size": 0.25},
        ),
        expect_agent="placement_agent",
        forbid_agents=("object_agent",),
    ),
    RoutingCase(
        name="creation_containment_routes_to_object",
        request="Add a small red sphere inside the cube.",
        scene=(
            {"id": "box-0", "type": "box", "pos": [-0.5, 1.3, -1.5], "color": [0, 0.4, 1], "size": 0.25},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent", "vision_agent"),
    ),
    RoutingCase(
        name="ahead_placement_needs_no_vision",
        request="Put a red sphere two meters ahead of me.",
        expect_agent="object_agent",
        forbid_agents=("vision_agent",),
    ),
    RoutingCase(
        name="remove_with_side_descriptor",
        request="Remove the pyramid on the left.",
        scene=(
            {"id": "pyramid-0", "type": "pyramid", "pos": [-1.0, 1.5, -2.2], "color": [0.4, 0.4, 0.4], "size": 0.2},
            {"id": "pyramid-1", "type": "pyramid", "pos": [1.0, 1.5, -2.2], "color": [0.4, 0.4, 0.4], "size": 0.2},
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent", "vision_agent"),
    ),
    RoutingCase(
        name="bare_create_no_invented_position",
        request="Okay. Make a sphere.",
        expect_agent="object_agent",
        instruction_forbids=("origin", "requested initial"),
    ),
    RoutingCase(
        name="bare_create_with_history",
        request="Make a blue sphere.",
        history=(
            ("Make a blue sphere.", "Created a blue sphere."),
            ("Make a green cube.", "The system has created a green cube in the scene."),
            ("Okay. Make a sphere.", "Created a sphere."),
        ),
        expect_agent="object_agent",
        instruction_forbids=("origin", "requested initial"),
    ),
    RoutingCase(
        name="add_verb_bare_create",
        request="Add a green cube.",
        expect_agent="object_agent",
        instruction_contains=("no position stated",),
        instruction_forbids=("origin",),
    ),
    RoutingCase(
        name="fragment_never_mutates",
        request="Fascinating.",
        history=(
            ("Make a red cube.", "Added a red cube."),
            ("Put a blue sphere above the green sphere.", "Blue sphere added above the green sphere."),
        ),
        forbid_agents=("object_agent", "placement_agent", "appearance_agent"),
    ),
    RoutingCase(
        name="create_new_anchored_routes_to_object",
        request="Put a yellow cube above the blue sphere.",
        scene=(
            {"id": "sphere-0", "type": "sphere", "pos": [0.0, 1.6, -1.5], "color": [0, 0, 1], "size": 0.1},
        ),
        # Known wart: a vision existence-check sometimes precedes the correct
        # delegation (latency-only; camera-off degrades gracefully). The old
        # zero-vision pass was prompt-vocabulary recall, not skill.
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
        instruction_contains=("above",),
    ),
    RoutingCase(
        name="row_creation_single_delegation",
        request="Create three red spheres in a row.",
        expect_agent="object_agent",
        instruction_contains=("three",),
    ),
    RoutingCase(
        name="bare_create_after_work_no_extra_move",
        request="Make a red cube.",
        scene=(
            {"id": "box-8", "type": "box", "pos": [0.1, 1.2, -0.9], "color": [0, 1, 1], "size": 0.1},
            {"id": "sphere-9", "type": "sphere", "pos": [-0.4, 1.4, -1.1], "color": [0, 0.8, 0], "size": 0.1},
        ),
        history=(
            ("Add a cyan cube.", "Added a cyan cube."),
            ("Make a green sphere.", "Created a green sphere."),
        ),
        expect_agent="object_agent",
        forbid_agents=("placement_agent",),
    ),
    RoutingCase(
        name="correction_never_creates",
        request="That's the wrong sphere.",
        scene=(
            {"id": "sphere-5", "type": "sphere", "pos": [-0.7, 1.2, -0.4], "color": [0, 0.8, 0], "size": 0.1},
            {"id": "sphere-6", "type": "sphere", "pos": [0.2, 1.2, -0.8], "color": [0, 0, 1], "size": 0.1},
            {"id": "box-3", "type": "box", "pos": [0.2, 1.5, -0.8], "color": [1, 0, 0], "size": 0.1},
        ),
        history=(
            ("Add a red cube above the green sphere.", "Added a red cube above the green sphere."),
        ),
        # Known wart: the supervisor sometimes consults vision/memory before
        # asking back; latency-only. The invariant is that nothing mutates.
        forbid_agents=("object_agent", "appearance_agent", "placement_agent"),
    ),
)


async def run_case(case: RoutingCase) -> bool:
    scene = harness.FakeScene.from_corpus_case(
        {"name": case.name, "scene": list(case.scene), "history": list(case.history), "user": case.request}
    )
    calls: list[tuple[str, str]] = []
    llm = make_llm(load_models_config(harness._CONFIG.models_yaml), "agent_llm")
    try:
        async with WorkflowBuilder() as builder:
            await scene.bind(builder)
            await builder.add_llm(
                LLMRef("scene_llm"),
                ModelsLLMConfig(
                    service=llm, model_name="xr-scene-agent", max_tokens=2048,
                    temperature=0.0,
                ),
            )
            for agent_name, description in _DESCRIPTIONS.items():
                await builder.add_function(
                    agent_name,
                    _FakeAgentConfig(agent_name=agent_name, description=description, recorder=calls),
                )
            reasoning = await builder.add_function(
                "supervisor_reasoning",
                ToolCallAgentWorkflowConfig(
                    llm_name=LLMRef("scene_llm"),
                    tool_names=[FunctionRef(name) for name in _DESCRIPTIONS],
                    system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
                    handle_tool_errors=True,
                    max_iterations=12,
                    max_empty_response_retries=1,
                ),
            )
            conversations = await builder.get_function_group("conversations")
            conversation_functions = await conversations.get_all_functions()
            recall = conversation_functions["conversations__recall_conversation"]
            recalled = await recall.ainvoke(RecallConversationRequest(participant_id="eval-user"))
            lines = [f"  {'User' if e.role == 'user' else 'Agent'}: {e.text}" for e in recalled.entries[-8:]]
            conversation = ("[Recent conversation]\n" + "\n".join(lines) + "\n\n") if lines else ""
            state = await scene.get_scene_state(harness.EmptyRequest())
            message = (
                f"Active participant: eval-user\nUtterance timestamp: 10000000\n"
                f"[SCENE OBJECTS]\n{state.model_dump_json()}\n\n{conversation}"
                f"User request: {case.request}"
            )
            try:
                output = await reasoning.ainvoke(message, to_type=str)
                reply = SceneReply(response=str(output or "Done."))
            except Exception as exc:
                reply = SceneReply(response=f"<workflow error: {exc}>")
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
    print(f"\nsupervisor routing: {sum(results)}/{len(results)} passed")
    if not all(results):
        raise SystemExit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
