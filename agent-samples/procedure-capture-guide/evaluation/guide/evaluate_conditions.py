# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate guide decision policy independently of camera perception.

Fixed observations and conditions exercise the production CLEF request. Results
preserve every attempt, including unavailable decisions, so prompt revisions can
be compared without hiding model errors. Existing servers are left running.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import yaml
from procedure_capture_guide_worker.guide import verification
from xr_ai_models import load_models_config, make_decision

_SAMPLE = Path(__file__).resolve().parents[2]
_CASES = Path(__file__).with_name("decision_cases.yaml")


async def evaluate(cases_path: Path, models_path: Path, output: Path, *, repeats: int = 3,
                   timeout_s: float = 15) -> dict:
    if repeats < 1 or timeout_s <= 0:
        raise ValueError("repeats and timeout must be positive")
    cases = yaml.safe_load(cases_path.read_text())["cases"]
    if not cases:
        raise ValueError("decision cases must not be empty")
    output.mkdir(parents=True, exist_ok=False)
    sources = [cases_path, models_path, Path(__file__), Path(verification.__file__),
               Path(verification.__file__).with_name("prompts") / "guide_condition.txt"]
    (output / "provenance.json").write_text(json.dumps({
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "cases": cases,
    }, indent=2) + "\n")
    decisions = make_decision(load_models_config(models_path), "decision")
    rows = []
    try:
        for trial in range(1, repeats + 1):
            for case in cases:
                accepted, error = None, None
                try:
                    async with asyncio.timeout(timeout_s):
                        accepted = await verification.condition_met(
                            decisions, condition=case["condition"], observation=case["observation"],
                            frame={}, step_id=case["id"], revision=trial, timeout_s=timeout_s,
                        )
                except (TimeoutError, ValueError, KeyError, RuntimeError, httpx.HTTPError) as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                row = {**case, "trial": trial, "accepted": accepted, "error": error,
                       "correct": accepted == case["expected"] if accepted is not None else None}
                rows.append(row)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    finally:
        await decisions.close()
    summary = {
        "attempts": len(rows), "correct": sum(r["correct"] is True for r in rows),
        "unavailable": sum(r["accepted"] is None for r in rows),
        "false_accepts": sum(r["accepted"] is True and not r["expected"] for r in rows),
        "false_rejects": sum(r["accepted"] is False and r["expected"] for r in rows),
        "passed": all(r["correct"] is True for r in rows),
        "coverage": "Decision policy on supplied descriptions; no image perception or guide execution.",
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_CASES)
    parser.add_argument("--models", type=Path, default=_SAMPLE / "yaml/models.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=15)
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.cases, args.models, args.output,
                                  repeats=args.repeats, timeout_s=args.timeout_s))
    raise SystemExit(0 if result["passed"] else 1)
