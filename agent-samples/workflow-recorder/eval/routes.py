# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate idle SOP routing against a serving model without executing commands."""

import argparse
import asyncio
import time
from pathlib import Path

import yaml
from workflow_recorder_worker.events import RECORDING_COMMAND, USER_QUERY_TOPIC
from xr_ai_models import load_models_config, make_llm
from xr_ai_sample_agents.conversation import _handback_reason
from xr_ai_sample_agents.front_end import ConversationApplication, QuickConversation

_SAMPLE = Path(__file__).resolve().parents[1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=_SAMPLE / "yaml/models.json")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("routes.yaml"))
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args()
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    application = ConversationApplication(
        name="sop_guide",
        description=(_SAMPLE / "worker/workflow_recorder_worker/prompts/top_level_route.txt").read_text(),
        query_topic=USER_QUERY_TOPIC,
        has_focus=lambda _pid: False,
        context=lambda _pid: "Workflow recording and approved SOP guides are available but idle.",
    )
    llm = make_llm(load_models_config(args.models), "llm")
    conversation = QuickConversation(None, None, llm=llm)  # type: ignore[arg-type]
    passed = 0
    try:
        for case in cases:
            started = time.perf_counter()
            if RECORDING_COMMAND.fullmatch(case["query"]):
                route = "control"
            else:
                context = application.context("eval")
                route = await conversation._route(
                    case["query"], (), context, (application.tool(),),
                )
                if route != application.name:
                    response = await conversation._decide_with_handoff(case["query"], (), context)
                    if (reason := _handback_reason(response)) is not None:
                        route = await conversation._route(
                            case["query"], (), context, (application.tool(),), handback=reason,
                        )
                        if route != application.name:
                            response = await conversation.decide(case["query"], app_context=context)
                    if route != application.name:
                        calls = response.tool_calls or ()
                        route = calls[0].name if len(calls) == 1 else "invalid" if calls else "direct"
            ok = route == case["route"]
            passed += ok
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            print(f"{'PASS' if ok else 'FAIL'} {case['name']}: {route}, expected {case['route']}, {elapsed_ms}ms")
    finally:
        await llm.close()
    print(f"Idle routes: {passed}/{len(cases)}. Focus and silence require the runtime regression tests.")
    if not cases or passed / len(cases) < args.min_score:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
