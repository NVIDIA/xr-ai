# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for the foreground monitoring assistant."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml
from background_monitoring_worker.foreground import (
    FOREGROUND_TOOL_DEFS,
    required_foreground_tool,
)
from xr_ai_models import ChatMessage, load_models_config, make_llm

_SAMPLE = Path(__file__).resolve().parents[1]


async def main() -> None:
    prompt = (
        _SAMPLE
        / "worker"
        / "background_monitoring_worker"
        / "prompts"
        / "foreground_prompt.txt"
    ).read_text(encoding="utf-8").strip()
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    llm = make_llm(
        load_models_config(_SAMPLE / "yaml" / "models.local.json"),
        "llm",
    )
    failures: list[str] = []
    try:
        for case in cases:
            messages = [
                ChatMessage(role="system", content=prompt),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {"request": case["query"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            ]
            required_tool = required_foreground_tool(case["query"])
            for _ in range(4):
                response = await llm.chat(
                    messages,
                    tools=FOREGROUND_TOOL_DEFS,
                    max_tokens=256,
                    temperature=0.0,
                    enable_thinking=False,
                )
                calls = response.tool_calls or []
                call = calls[0] if calls else None
                if required_tool is None or (call and call.name == required_tool):
                    break
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=list(calls),
                    )
                )
                for rejected in calls:
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps(
                                {
                                    "error": "wrong_route",
                                    "required_tool": required_tool,
                                },
                                separators=(",", ":"),
                            ),
                            tool_call_id=rejected.id,
                        )
                    )
                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"Call {required_tool} for the original request.",
                    )
                )
            else:
                call = None
            actual_tool = call.name if call else None
            actual_target = json.loads(call.arguments).get("target") if call else None
            expected_tool = case["expected_tool"]
            expected_target = case.get("expected_target")
            passed = (actual_tool, actual_target) == (expected_tool, expected_target)
            label = "PASS" if passed else "FAIL"
            print(
                f"{label} {case['name']}: "
                f"tool={actual_tool!r} target={actual_target!r}"
            )
            if not passed:
                print(f"  content={response.content!r}")
                failures.append(
                    f"{case['name']}: expected {(expected_tool, expected_target)!r}, "
                    f"received {(actual_tool, actual_target)!r}"
                )
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
