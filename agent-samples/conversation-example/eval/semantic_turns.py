# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay incremental ASR snapshots through real Clef or a labeled plumbing fixture."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from conversation_example_worker._semantic_turns import _QUESTIONS, _Policy, _TurnController
from conversation_example_worker.config import load_config
from xr_ai_models import (
    DecisionAnswer,
    DecisionResponse,
    load_models_config_from_dict,
    make_decision,
)


class _FixtureDecisions:
    """Scripted labels exercise orchestration only; never evidence of model quality."""

    labels: dict

    async def decide(self, state, questions, **kwargs):
        answers = {}
        for key, question in questions.items():
            choice = self.labels[key]
            answers[key] = DecisionAnswer(choice, 1.0, {k: float(k == choice) for k in question.criteria})
        return DecisionResponse(answers, "scripted-plumbing-only", {}, {})

    async def close(self):
        pass


async def _run(args) -> int:
    policy = load_config(args.config).semantic_turns_policy
    if args.fixture:
        decisions = _FixtureDecisions()
    else:
        models = load_models_config_from_dict(
            {
                "models": {
                    "clef": {
                        "category": "decision",
                        "kind": "systemone",
                        "model_name": "Cloudflare/clef-flash",
                        "base_url": args.clef_url,
                    }
                }
            }
        )
        decisions = make_decision(models, "clef")
        if not await decisions.health():
            await decisions.close()
            raise RuntimeError("Clef not healthy; start the existing service or supply --clef-url")
    controller = _TurnController(decisions, policy)
    failures = 0
    checked = 0
    try:
        if args.classify:
            return await _classify(decisions, args.events, policy)
        for number, line in enumerate(args.events.read_text().splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event["kind"]
            participant, epoch = event.get("participant", "user"), event.get("epoch", "track-1")
            started = time.perf_counter()
            actions = []
            if kind == "open":
                controller.open(participant, epoch)
            elif kind == "reset":
                controller.reset(participant, epoch)
            elif kind == "text":
                actions = controller.feed(
                    participant,
                    epoch,
                    segment=event["segment"],
                    revision=event.get("revision", 0),
                    text=event["text"],
                    now=event["at"],
                )
            elif kind == "poll":
                if args.fixture:
                    decisions.labels = event["labels"]
                actions = await controller.poll(participant, epoch, now=event["at"])
            elif kind == "response":
                actions = controller.response(
                    participant,
                    epoch,
                    event["request_id"],
                    event["stage"],
                    effects_started=event.get("effects_started", False),
                    now=event.get("at"),
                )
            else:
                raise ValueError(f"unknown event kind on line {number}: {kind}")
            record = {
                "line": number,
                "kind": kind,
                "actions": [asdict(a) for a in actions],
                "classification_and_policy_ms": round((time.perf_counter() - started) * 1000, 3),
                "backend": "scripted-plumbing-only" if args.fixture else "real-clef",
            }
            if "expect" in event:
                checked += 1
                record["passed"] = [a.kind for a in actions] == event["expect"]
                failures += not record["passed"]
            print(json.dumps(record), flush=True)
    finally:
        controller.close()
        await decisions.close()
    print(json.dumps({"checked": checked, "failures": failures}), file=sys.stderr)
    return int(failures > 0)


async def _classify(decisions, path: Path, policy: _Policy) -> int:
    """Score independent human-labeled inputs; no fixture backend is permitted."""
    failures = 0
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        state = {
            "candidate": case["candidate"],
            "previous_request": case.get("previous_request", ""),
            "response_stage": case.get("response_stage", "none"),
            "effects_started": case.get("effects_started", False),
            "gap_s": case.get("gap_s"),
        }
        started = time.perf_counter()
        result = await decisions.decide(state, _QUESTIONS, timeout=policy.decision_timeout_s)
        labels = {key: answer.choice for key, answer in result.answers.items()}
        passed = all(labels[key] in allowed for key, allowed in case["acceptable"].items())
        failures += not passed
        count += 1
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "labels": labels,
                    "confidence": {key: answer.confidence for key, answer in result.answers.items()},
                    "probabilities": {key: dict(answer.probabilities) for key, answer in result.answers.items()},
                    "model": result.model,
                    "backend": "real-clef",
                    "passed": passed,
                    "decision_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            ),
            flush=True,
        )
    print(json.dumps({"checked": count, "failures": failures}), file=sys.stderr)
    return int(failures > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, nargs="?", help="JSONL transcript, clock, and output lifecycle events")
    parser.add_argument(
        "--clef-url",
        default="http://127.0.0.1:8120",
        help="Replay/classification endpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "yaml/conversation_example_worker.yaml",
        help="Conversation worker YAML supplying semantic-turn policy settings",
    )
    parser.add_argument("--fixture", action="store_true", help="Use scripted labels; tests plumbing, not Clef")
    parser.add_argument("--classify", action="store_true", help="Score independent cases using real Clef")
    args = parser.parse_args()
    if args.fixture and args.classify:
        parser.error("--classify requires real Clef; fixture labels cannot establish model quality")
    if args.events is None:
        parser.error("supply an events file")
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
