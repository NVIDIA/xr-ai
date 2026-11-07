# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay alert compilation and record outputs for semantic review, without starting watches."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml
from xr_ai_models import ChatResponse, LLMService, ModelsConfig, load_models_config, make_llm
from xr_ai_routing import AgentExchange

_SAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from conversation_example_worker._alert_agent import _COMPILE_TIMEOUT_SECS  # noqa: E402
from conversation_example_worker._alert_compile import _MAX_TOKENS, compile_alert  # noqa: E402
from conversation_example_worker.app import _alert_compile_models  # noqa: E402


def load_cases() -> list[dict[str, Any]]:
    return yaml.safe_load((_SAMPLE / "eval/alert-compile-cases.yaml").read_text(encoding="utf-8"))


class _Recorder:
    """Capture the production request; optional overrides isolate generation limits."""

    def __init__(self, llm: LLMService, budget: int | None, max_tokens: int | None) -> None:
        self.llm = llm
        self.budget = budget
        self.max_tokens = max_tokens
        self.response: ChatResponse | None = None
        self.request: dict[str, Any] = {}

    async def chat(self, messages, **kwargs) -> ChatResponse:
        if self.budget is not None:
            kwargs["thinking_budget"] = self.budget
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        self.request = {"messages": [asdict(message) for message in messages], "options": kwargs}
        self.response = await self.llm.chat(messages, **kwargs)
        return self.response


async def replay(
    llm: LLMService,
    case: dict[str, Any],
    *,
    budget: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    recorder = _Recorder(llm, budget, max_tokens)
    started = time.perf_counter()
    watch = None
    timed_out = False
    try:
        async with asyncio.timeout(_COMPILE_TIMEOUT_SECS):
            watch = await compile_alert(
                recorder,
                case["query"],
                history=tuple(AgentExchange(item["user"], item["assistant"]) for item in case.get("history", [])),
                context=case.get("context", ""),
            )
    except TimeoutError:
        timed_out = True
    seconds = time.perf_counter() - started
    response = recorder.response
    actual_status = "timeout" if timed_out else "ready" if watch else "invalid"
    if response is not None and watch is None:
        text = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict) and body.get("status") == "unclear":
            actual_status = "unclear"
    return {
        "case": case["name"],
        "expected_status": case["expected_status"],
        "expected_condition": case.get("expected_condition"),
        "notes": case.get("notes"),
        "actual_status": actual_status,
        "status_matches": actual_status == case["expected_status"],
        "semantic_review_required": actual_status == "ready",
        "watch": asdict(watch) if watch else None,
        "seconds": seconds,
        "content": response.content if response else None,
        "finish_reason": response.finish_reason if response else None,
        "usage": response.raw.get("usage") if response else None,
        "request": recorder.request,
        "messages_sha256": hashlib.sha256(
            json.dumps(recorder.request.get("messages"), sort_keys=True).encode()
        ).hexdigest(),
    }


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


async def _run(args: argparse.Namespace) -> None:
    cases = [case for case in load_cases() if not args.case or case["name"] in args.case]
    models = _alert_compile_models(load_models_config(args.models))
    budgets = args.thinking_budgets or [None]
    # Alternate paired budgets to reduce ordering effects; keep input and seed fixed within a pair.
    with args.output.open("x", encoding="utf-8") as output:
        for repeat in range(args.repeat):
            seed = args.seed + repeat
            spec = models.llm("alert_compile_llm")
            spec = replace(spec, adapter=replace(spec.adapter, default_extras={**spec.default_extras, "seed": seed}))
            llm = make_llm(ModelsConfig({**models.entries, "alert_compile_llm": spec}), "alert_compile_llm")
            try:
                for case in cases:
                    for budget in budgets if repeat % 2 == 0 else list(reversed(budgets)):
                        row = await replay(llm, case, budget=budget, max_tokens=args.max_tokens)
                        row.update(
                            seed=seed, model=spec.adapter.model_name, temperature=spec.default_extras.get("temperature")
                        )
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        print(
                            f"{case['name']} seed={seed} budget={row['request']['options']['thinking_budget']} "
                            f"status={row['actual_status']} seconds={row['seconds']:.2f}",
                            flush=True,
                        )
            finally:
                await llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=_SAMPLE / "yaml/models.json")
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSONL file; existing files are not overwritten."
    )
    parser.add_argument("--case", action="append", choices=[case["name"] for case in load_cases()])
    parser.add_argument("--repeat", type=_positive, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thinking-budgets", type=_positive, nargs="+", help="Override reasoning caps for paired trials."
    )
    parser.add_argument(
        "--max-tokens", type=_positive, help="Fixed total-output cap; omitted uses the production limit."
    )
    args = parser.parse_args()
    total_limit = args.max_tokens if args.max_tokens is not None else _MAX_TOKENS
    if args.thinking_budgets and total_limit <= max(args.thinking_budgets):
        parser.error("--max-tokens must exceed every thinking budget to leave answer room")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
