# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate first-turn acknowledgements against a running sample LLM.

Run from the repository root with a serving LLM::

    uv run --project agent-sdk/xr-ai-voice --with pyyaml \
        python agent-sdk/xr-ai-voice/eval/acknowledgements.py
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import yaml
from xr_ai_models import load_models_config, make_llm
from xr_ai_voice import VoiceOutput, VoiceTurnController
from xr_ai_voice._coordination import _acknowledge_if_needed

_ROOT = Path(__file__).resolve().parents[3]
_CASES = Path(__file__).with_name("acknowledgement_cases.yaml")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-config",
        type=Path,
        default=_ROOT / "agent-samples/xr-render-demo/yaml/models.json",
    )
    parser.add_argument("--model-key", default="llm")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    cases = yaml.safe_load(_CASES.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["name"] in args.case]
    if not cases:
        raise SystemExit("no acknowledgement evaluation cases selected")

    llm = make_llm(load_models_config(args.models_config), args.model_key)
    passed = 0
    elapsed_ms: list[float] = []
    try:
        for case in cases:
            outputs: list[VoiceOutput] = []

            async def publish(output: VoiceOutput) -> None:
                outputs.append(output)

            controller = VoiceTurnController(turn_id=case["name"], timestamp_us=None, publish=publish)
            started = time.perf_counter()
            await _acknowledge_if_needed(controller, llm, case["query"], context=case["context"])
            elapsed_ms.append((time.perf_counter() - started) * 1000)
            answer = outputs[0].text if outputs else ""
            normalized = answer.casefold()
            expected = case["acknowledge"]
            ok = bool(answer) == expected
            if answer:
                ok = ok and not any(word.casefold() in normalized for word in case.get("none_of", ()))
                if case.get("any_of"):
                    ok = ok and any(word.casefold() in normalized for word in case["any_of"])
            passed += ok
            print(
                f"{'PASS' if ok else 'FAIL'} {case['name']} "
                f"expected={expected} actual={bool(answer)} {elapsed_ms[-1]:.0f}ms: {answer}",
                flush=True,
            )
    finally:
        await llm.close()
    score = passed / len(cases)
    print(f"acknowledgements: {passed}/{len(cases)} ({score:.1%}); mean={sum(elapsed_ms) / len(elapsed_ms):.0f}ms")
    if score < args.min_score:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
