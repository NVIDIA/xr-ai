# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for the foreground monitoring assistant."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from lab_instrument_monitoring_worker.foreground import FOREGROUND_TOOL_DEFS
from xr_ai_models import ChatMessage, load_models_config, make_llm

_SAMPLE = Path(__file__).resolve().parents[1]


async def main() -> None:
    prompt = (
        (_SAMPLE / "worker" / "lab_instrument_monitoring_worker" / "prompts" / "foreground_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
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
                ChatMessage(role="user", content=case["query"]),
            ]
            response = await llm.chat(
                messages,
                tools=FOREGROUND_TOOL_DEFS,
                max_tokens=256,
                temperature=0.0,
                enable_thinking=False,
            )
            calls = response.tool_calls or []
            call = calls[0] if calls else None
            actual_tool = call.name if call else None
            expected_tool = case["expected_tool"]
            passed = actual_tool == expected_tool
            label = "PASS" if passed else "FAIL"
            print(f"{label} {case['name']}: tool={actual_tool!r}")
            if not passed:
                print(f"  content={response.content!r}")
                failures.append(f"{case['name']}: expected {expected_tool!r}, received {actual_tool!r}")
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
