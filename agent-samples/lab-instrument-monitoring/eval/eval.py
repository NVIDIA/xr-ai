# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for the foreground monitoring assistant."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import yaml
from lab_instrument_monitoring_worker.foreground import (
    _ROUTER_REASONING_GUIDANCE,
    FOREGROUND_TOOL_DEFS,
    FOREGROUND_TOOL_REQUEST_MODELS,
)
from xr_ai_models import ChatMessage, ChatResponse, LLMService, load_models_config, make_llm

_SAMPLE = Path(__file__).resolve().parents[1]


def _models_config() -> Path:
    override = os.environ.get("XR_AI_EVAL_MODELS_CONFIG")
    return Path(override) if override else _SAMPLE / "yaml" / "models.json"


async def _adaptive_response(
    llm: LLMService,
    prompt: str,
    query: str,
    *,
    turn_id: str,
) -> tuple[ChatResponse, bool]:
    transcript = (
        ChatMessage(
            role="system",
            content=f"{prompt}\n\n{_ROUTER_REASONING_GUIDANCE}",
        ),
        ChatMessage(role="user", content=query),
    )
    response = await llm.chat(
        transcript,
        tools=FOREGROUND_TOOL_DEFS,
        max_tokens=1536,
        temperature=0.0,
        enable_thinking=True,
        thinking_budget=1024,
    )
    return response, True


async def main() -> None:
    prompt = (
        (_SAMPLE / "worker" / "lab_instrument_monitoring_worker" / "prompts" / "foreground_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case names; omit to run all")
    args = parser.parse_args()
    wanted = set(args.cases)
    cases = [case for case in cases if not wanted or case["name"] in wanted]
    if not cases:
        raise SystemExit(f"unknown cases: {args.cases}")
    llm = make_llm(
        load_models_config(_models_config()),
        "llm",
    )
    failures: list[str] = []
    try:
        for case in cases:
            response, reasoning_enabled = await _adaptive_response(
                llm,
                prompt,
                case["query"],
                turn_id=f"lab-eval:{case['name']}",
            )
            calls = response.tool_calls or []
            expected_tool = case["expected_tool"]
            actual_tools = [call.name for call in calls]
            expected_tools = [] if expected_tool is None else [expected_tool]
            errors: list[str] = []
            for call in calls:
                request_model = FOREGROUND_TOOL_REQUEST_MODELS.get(call.name)
                if request_model is None:
                    errors.append(f"unknown tool {call.name!r}")
                    continue
                try:
                    request_model.model_validate_json(call.arguments)
                except ValueError as exc:
                    errors.append(f"invalid {call.name!r} arguments: {exc}")
            passed = actual_tools == expected_tools and not errors
            label = "PASS" if passed else "FAIL"
            print(
                f"{label} {case['name']}: tools={actual_tools!r} "
                f"reasoning={reasoning_enabled}"
            )
            if not passed:
                print(f"  content={response.content!r}")
                failures.append(
                    f"{case['name']}: expected {expected_tools!r}, received {actual_tools!r}; errors={errors!r}"
                )
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
