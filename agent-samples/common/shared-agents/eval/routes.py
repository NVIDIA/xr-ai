# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated top-level and conversation route evaluation against a serving LLM."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import yaml
from xr_ai_models import ToolDef, load_models_config, make_llm
from xr_ai_sample_agents import ConversationExchange, QuickConversation
from xr_ai_sample_agents.conversation import _handback_reason

_ROOT = Path(__file__).resolve().parents[4]
_CASES = Path(__file__).with_name("routes.yaml")
_ROUTES = {
    "xr": (
        "xr_scene",
        _ROOT / "agent-samples/xr-render-demo/worker/xr_render_demo_worker/prompts/top_level_route.txt",
        "A virtual XR scene may be available.",
    ),
    "tea": (
        "tea_guide",
        _ROOT / "agent-samples/tea-making-sample/worker/tea_making_worker/prompts/top_level_route.txt",
        "Tea-making guidance is available but idle.",
    ),
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=_ROOT / "agent-samples/simple-vlm-example/yaml/models.json",
    )
    parser.add_argument("--model-key", default="llm")
    parser.add_argument("--sample", choices=(*_ROUTES, "both"), default="both")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--corpus", type=Path, default=_CASES)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    corpus = yaml.safe_load(args.corpus.read_text(encoding="utf-8"))
    extra_apps = corpus.get("applications", {}) if isinstance(corpus, dict) else {}
    cases = corpus["cases"] if isinstance(corpus, dict) else corpus
    cases = [
        case
        for case in cases
        if (args.sample == "both" or case.get("sample") == args.sample)
        and (not args.case or case["name"] in args.case)
    ]
    if not cases:
        raise SystemExit("no route cases selected")

    llm = make_llm(load_models_config(args.models), args.model_key)
    conversation = QuickConversation(  # type: ignore[arg-type]
        None, None, llm=llm, thinking_budget=args.thinking_budget
    )
    passed = 0
    elapsed = []
    try:
        for case in cases:
            if "catalog" in case:
                catalog = case["catalog"]
                descriptions = {name: extra_apps[name] for name in catalog}
                context = case.get("context", "")
            else:
                name, description_file, context = _ROUTES[case["sample"]]
                descriptions = {name: description_file.read_text(encoding="utf-8").strip()}
            apps = tuple(
                ToolDef(
                    name=name,
                    description=description,
                    parameters={"type": "object", "properties": {}, "additionalProperties": False},
                )
                for name, description in descriptions.items()
            )
            history = tuple(ConversationExchange(**turn) for turn in case.get("history", ()))
            started = time.perf_counter()
            first_route = await conversation._route(case["query"], history, context, apps)
            response = None
            route = first_route
            if first_route == "conversation":
                response = await conversation._decide_with_handoff(case["query"], history, context)
                if (reason := _handback_reason(response)) is not None:
                    route = await conversation._route(
                        case["query"], history, context, apps, handback=reason
                    )
                    if route == "conversation":
                        response = await conversation.decide(
                            case["query"], history=history, app_context=context
                        )
                if route == "conversation":
                    calls = response.tool_calls or ()
                    route = calls[0].name if len(calls) == 1 else "direct"
            duration_ms = round((time.perf_counter() - started) * 1000)
            elapsed.append(duration_ms)
            answer = response.content if response is not None else ""
            clarified = "?" in answer if case.get("clarify") else True
            answered = case.get("answer_contains", "").lower() in answer.lower()
            ok = route == case["route"] and clarified and answered
            passed += ok
            clarification = f" clarified={clarified}" if case.get("clarify") else ""
            answer_match = f" answered={answered}" if case.get("answer_contains") else ""
            print(
                f"{'PASS' if ok else 'FAIL'} {case.get('sample', 'synthetic')}:{case['name']} "
                f"first={first_route} route={route} expected={case['route']} "
                f"{duration_ms}ms{clarification}{answer_match} answer={answer[:100]!r}",
                flush=True,
            )
    finally:
        await llm.close()
    print(
        f"hierarchical routing: {passed}/{len(cases)} ({passed / len(cases):.1%}); "
        f"mean={sum(elapsed) / len(elapsed):.0f}ms"
    )
    if passed / len(cases) < args.min_score:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
