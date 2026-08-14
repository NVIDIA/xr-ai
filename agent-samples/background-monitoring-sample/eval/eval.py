# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model routing eval for the foreground monitoring assistant."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from background_monitoring_worker.foreground import FOREGROUND_TOOL_DEFS
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
            response = await llm.chat(
                [
                    ChatMessage(role="system", content=prompt),
                    ChatMessage(role="user", content=case["query"]),
                ],
                tools=FOREGROUND_TOOL_DEFS,
                max_tokens=256,
                temperature=0.0,
                enable_thinking=False,
            )
            actual = response.tool_calls[0].name if response.tool_calls else None
            expected = case["expected_tool"]
            passed = actual == expected
            print(f"{'PASS' if passed else 'FAIL'} {case['name']}: {actual!r}")
            if not passed:
                failures.append(
                    f"{case['name']}: expected {expected!r}, received {actual!r}"
                )
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
