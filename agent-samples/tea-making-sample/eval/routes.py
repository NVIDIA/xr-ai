# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the production foreground voice agents against the configured model."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict
from xr_ai_models import load_models_config, make_llm
from xr_ai_nat.events import EventDispatcher

_SAMPLE = Path(__file__).parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents import AgentRegistry  # noqa: E402
from tea_making_worker.agents.factory import add_guidance_llm  # noqa: E402
from tea_making_worker.applications.compose import build_applications  # noqa: E402
from tea_making_worker.applications.context import (  # noqa: E402
    ApplicationContextFunctionsConfig,
    add_context_query,
)
from tea_making_worker.applications.events import BACKGROUND_FACT  # noqa: E402
from tea_making_worker.applications.manager.runtime import ApplicationOwnership  # noqa: E402
from tea_making_worker.applications.manager.spec import load_application_catalog  # noqa: E402
from tea_making_worker.applications.manager.turn import ApplicationTurn  # noqa: E402
from tea_making_worker.functions import add_workflow_functions  # noqa: E402
from tea_making_worker.runtime.scope import current_invocation, invocation_scope  # noqa: E402
from tea_making_worker.runtime.state import Session, SessionStore  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


class QuickRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class _QuickConfig(FunctionBaseConfig, name="tea_voice_route_eval_quick_tool"):
    tool: str


class _Output:
    async def publish(self, _participant_id, _producer, output, **_kwargs) -> str:
        return output.text


@register_function(config_type=_QuickConfig)
async def _quick_tool(config: _QuickConfig, _builder: Builder):
    async def invoke(request: QuickRequest) -> str:
        return f"No live {config.tool} result is available in this routing evaluation."

    yield FunctionInfo.from_fn(
        invoke,
        description=f"Read-only {config.tool} quick command; never changes foreground or workflow state.",
    )


def _expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [dict(case) for case in config["routes"]]
    matrix = config["route_state_matrix"]
    cases.extend({**case, "active": False} for case in matrix["idle_cases"])
    cases.extend(
        {**case, "active": True, "step": step}
        for step in matrix["active_steps"]
        for case in matrix["active_cases"]
    )
    return cases


async def evaluate(models_path: Path, cases_path: Path) -> int:
    workflow = load_workflow(_SAMPLE / "yaml" / "workflow.yaml")
    application_spec = load_application_catalog(_SAMPLE / "yaml" / "applications.yaml")
    application_runtime = ApplicationOwnership(application_spec)
    store = SessionStore(workflow)
    registry = AgentRegistry(workflow)
    cases = _expand_cases(yaml.safe_load(cases_path.read_text(encoding="utf-8")))
    llm = make_llm(load_models_config(models_path), "agent_llm")

    failures = 0

    async def notice(*_args) -> None:
        return None

    try:
        async with WorkflowBuilder() as builder:
            await add_workflow_functions(
                builder,
                store=store,
                application_ownership=application_runtime,
            )
            context_group = await builder.add_function_group(
                "context_store",
                ApplicationContextFunctionsConfig(),
            )
            context_functions = await context_group.get_all_functions()
            events = EventDispatcher()
            events.subscribe(
                BACKGROUND_FACT,
                subscriber_id="context.recorder",
                function=context_functions["context_store__record"],
            )
            await add_context_query(builder, context_functions["context_store__query"])
            quick_tools = {"current_view", "rag_lookup"}
            quick_tools.update(tool for step in workflow.steps.values() for tool in step.voice.tools)
            for tool in sorted(quick_tools):
                await builder.add_function(tool, _QuickConfig(tool=tool))
            llm_ref = await add_guidance_llm(builder, llm)
            await registry.build_foreground(builder, llm_ref)
            current_view = await builder.get_function("current_view")
            applications = await build_applications(
                builder,
                llm_ref=llm_ref,
                spec=application_spec,
                ownership=application_runtime,
                tea=registry,
                current_view=current_view,
                events=events,
                output=_Output(),  # type: ignore[arg-type]
            )
            for index, case in enumerate(cases):
                session = store.get(f"route-eval-{index}")
                if case.get("active", True):
                    _activate_at(store, session, str(case.get("step", workflow.start_step)))
                    application_runtime.capture(session, "tea")
                before_step = session.step_id
                with invocation_scope(session, f"route-eval-{index}"):
                    await applications.manager.function.ainvoke(
                        ApplicationTurn(request=str(case["utterance"])),
                        to_type=str,
                    )
                    actual = current_invocation().route_operation or "answer"
                expected = _operation(str(case["expected_tool"]))
                passed = actual == expected
                if "expected_advanced" in case:
                    assert before_step is not None
                    next_step = workflow.step(before_step).next_step
                    advanced = session.step_id == next_step and session.active == (next_step is not None)
                    passed = passed and advanced == case["expected_advanced"]
                if "expected_instruction_terms" in case:
                    instruction = applications.change_watch._states[session.participant_id].instruction
                    passed = passed and all(
                        str(term).lower() in instruction.lower()
                        for term in case["expected_instruction_terms"]
                    )
                failures += not passed
                print(f"{'PASS' if passed else 'FAIL'} {case['utterance']!r}: {actual} (expected {expected})")
                for application in (
                    applications.change_watch,
                    applications.transcript,
                    applications.video_log,
                ):
                    await application.release(session)
    finally:
        await llm.close()
    print(f"{len(cases) - failures}/{len(cases)} routes passed")
    return failures


def _activate_at(store: SessionStore, session: Session, step_id: str) -> None:
    store.start(session)
    while session.step_id != step_id:
        if session.step_id is None:
            raise ValueError(f"step {step_id!r} is not reachable")
        store.advance(session, skip=True)


def _operation(tool: str) -> str:
    if tool.startswith("workflow__"):
        return tool.removeprefix("workflow__")
    if "__" in tool:
        return tool.replace("__", ".")
    return tool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=_SAMPLE / "eval" / "cases.yaml")
    args = parser.parse_args()
    raise SystemExit(bool(asyncio.run(evaluate(args.models, args.cases))))


if __name__ == "__main__":
    main()
