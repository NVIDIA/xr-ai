# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the production voice router against the configured agent model."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_models import load_models_config, make_llm

_SAMPLE = Path(__file__).parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents import AgentRegistry  # noqa: E402
from tea_making_worker.functions import add_workflow_functions  # noqa: E402
from tea_making_worker.runtime.scope import current_invocation, invocation_scope  # noqa: E402
from tea_making_worker.runtime.state import SessionStore  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


async def evaluate(models_path: Path, cases_path: Path) -> int:
    workflow = load_workflow(_SAMPLE / "yaml" / "workflow.yaml")
    store = SessionStore(workflow)
    registry = AgentRegistry(workflow)
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))["routes"]
    llm = make_llm(load_models_config(models_path), "agent_llm")

    async def answer_step(*_args) -> str:
        return "step route"

    async def answer_general(*_args) -> str:
        return "general route"

    failures = 0
    try:
        async with WorkflowBuilder() as builder:
            await add_workflow_functions(
                builder,
                store=store,
                answer_step=answer_step,
                answer_tea=registry.route_tea,
                answer_general=answer_general,
            )
            await registry.build_router(builder, llm)
            for index, case in enumerate(cases):
                session = store.get(f"route-eval-{index}")
                if case.get("active", True):
                    store.start(session)
                    session.step_id = case.get("step", workflow.start_step)
                with invocation_scope(session, f"route-eval-{index}"):
                    await registry.route(session, str(case["utterance"]), f"route-eval-{index}")
                    actual = current_invocation().route_operation
                expected = str(case["expected_tool"]).removeprefix("workflow__")
                passed = actual == expected
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
