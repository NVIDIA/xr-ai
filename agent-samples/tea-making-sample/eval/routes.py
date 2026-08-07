# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the production foreground voice agents against the configured model."""

from __future__ import annotations

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

_SAMPLE = Path(__file__).parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents import AgentRegistry  # noqa: E402
from tea_making_worker.functions import add_workflow_functions  # noqa: E402
from tea_making_worker.runtime.scope import current_invocation, invocation_scope  # noqa: E402
from tea_making_worker.runtime.state import SessionStore  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


class _QuickRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class _QuickConfig(FunctionBaseConfig, name="tea_voice_route_eval_quick_tool"):
    tool: str


@register_function(config_type=_QuickConfig)
async def _quick_tool(config: _QuickConfig, _builder: Builder):
    async def invoke(request: _QuickRequest) -> str:
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
    store = SessionStore(workflow)
    registry = AgentRegistry(workflow)
    cases = _expand_cases(yaml.safe_load(cases_path.read_text(encoding="utf-8")))
    llm = make_llm(load_models_config(models_path), "agent_llm")

    failures = 0
    try:
        async with WorkflowBuilder() as builder:
            await add_workflow_functions(builder, store=store)
            quick_tools = {"current_view", "rag_lookup"}
            quick_tools.update(tool for step in workflow.steps.values() for tool in step.voice.tools)
            for tool in sorted(quick_tools):
                await builder.add_function(tool, _QuickConfig(tool=tool))
            await registry.build_foreground(builder, llm)
            for index, case in enumerate(cases):
                session = store.get(f"route-eval-{index}")
                if case.get("active", True):
                    store.start(session)
                    session.step_id = case.get("step", workflow.start_step)
                before_step = session.step_id
                with invocation_scope(session, f"route-eval-{index}"):
                    await registry.route(session, str(case["utterance"]), f"route-eval-{index}")
                    actual = current_invocation().route_operation or "answer"
                expected = str(case["expected_tool"]).removeprefix("workflow__")
                passed = actual == expected
                if "expected_advanced" in case:
                    assert before_step is not None
                    next_step = workflow.step(before_step).next_step
                    advanced = session.step_id == next_step and session.active == (next_step is not None)
                    passed = passed and advanced == case["expected_advanced"]
                failures += not passed
                print(f"{'PASS' if passed else 'FAIL'} {case['utterance']!r}: {actual} (expected {expected})")
    finally:
        await llm.close()
    print(f"{len(cases) - failures}/{len(cases)} routes passed")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=_SAMPLE / "eval" / "cases.yaml")
    args = parser.parse_args()
    raise SystemExit(bool(asyncio.run(evaluate(args.models, args.cases))))


if __name__ == "__main__":
    main()
