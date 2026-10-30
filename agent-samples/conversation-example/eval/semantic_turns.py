# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay early advice plus ordinary ASR finalization; score actual admission behavior."""

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
from conversation_example_worker._semantic_voice import _ClefVoicePolicy
from conversation_example_worker.config import load_config
from xr_ai_models import (
    DecisionAnswer,
    DecisionResponse,
    load_models_config,
    load_models_config_from_dict,
    make_decision,
)
from xr_ai_voice import TranscriptUpdate


class _Fixture:
    choice = "yes"
    confidence = 0.95

    async def decide(self, state, questions, **kwargs):
        return DecisionResponse(
            {"ready": DecisionAnswer(self.choice, self.confidence, {self.choice: self.confidence})},
            "scripted-plumbing-only",
            {},
            {},
        )

    async def close(self):
        pass


class _Probe:
    def __init__(self, service):
        self.service = service
        self.last = None
        self.error = None

    async def decide(self, *args, **kwargs):
        try:
            result = await self.service.decide(*args, **kwargs)
            self.last = result.answers["ready"]
            return result
        except Exception as exc:
            self.error = type(exc).__name__
            raise


async def _run(args):
    config = load_config(args.config)
    if args.fixture:
        service = _Fixture()
    else:
        models = load_models_config(config.models_config)
        role = "router"
        if args.clef_url:
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
            role = "clef"
        service = make_decision(models, role)
        if not await service.health():
            await service.close()
            raise RuntimeError("Clef is not healthy; start the existing service")
    checked = failures = 0
    backend = "scripted-plumbing-only" if args.fixture else "real-clef"
    try:
        if args.classify:
            for line in args.events.read_text().splitlines():
                if not line.strip():
                    continue
                case = json.loads(line)
                probe = _Probe(service)
                policy = _ClefVoicePolicy(probe, config.semantic_turns_policy)
                started = time.perf_counter()
                policy.update(TranscriptUpdate("case", "source", 0, 0, case["candidate"], 0))
                early = await policy.decide("case")
                early_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                policy.update(TranscriptUpdate("case", "source", 0, 1, case["candidate"], 0, True))
                final = await policy.decide("case")
                final_ms = (time.perf_counter() - started) * 1000
                actions = early + final
                passed = bool(actions) == case["expected_final"] and len(actions) <= 1
                if not case.get("allow_early", True):
                    passed &= not early
                if actions:
                    passed &= actions[0].text == case["candidate"]
                checked += 1
                failures += not passed
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "backend": backend,
                            "early_submitted": bool(early),
                            "final_submitted": bool(final),
                            "choice": probe.last.choice if probe.last else None,
                            "score": probe.last.confidence if probe.last else None,
                            "model_error": probe.error,
                            "early_threshold": config.semantic_turns_policy.min_confidence,
                            "early_ms": round(early_ms, 3),
                            "final_ms": round(final_ms, 3),
                            "passed": bool(passed),
                        }
                    ),
                    flush=True,
                )
        else:
            policy = _ClefVoicePolicy(service, config.semantic_turns_policy)
            epochs = {}
            for number, line in enumerate(args.events.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                pid = event.get("participant_id", "user")
                kind = event["kind"]
                actions = []
                if kind == "open":
                    policy.reset(pid)
                    epochs[pid] = event["epoch"]
                elif kind == "text":
                    policy.update(
                        TranscriptUpdate(
                            pid,
                            epochs[pid],
                            event["segment"],
                            event["revision"],
                            event["text"],
                            event.get("timestamp_us", 0),
                            event.get("final", False),
                        )
                    )
                elif kind == "poll":
                    if args.fixture:
                        service.choice = event.get("choice", "yes")
                        service.confidence = event.get("confidence", 0.95)
                    actions = await policy.decide(pid)
                elif kind == "reset":
                    policy.reset(pid)
                else:
                    raise ValueError(f"unknown event: {kind}")
                record = {"line": number, "backend": backend, "actions": [asdict(a) for a in actions]}
                if "expect" in event:
                    actual = [{"kind": a.kind, "text": a.text} for a in actions]
                    record["passed"] = actual == event["expect"]
                    checked += 1
                    failures += not record["passed"]
                print(json.dumps(record), flush=True)
    finally:
        await service.close()
    print(json.dumps({"checked": checked, "failures": failures}), file=sys.stderr)
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "yaml/conversation_example_worker.yaml"
    )
    parser.add_argument("--clef-url", help="Optional override of the configured Clef endpoint")
    parser.add_argument("--fixture", action="store_true", help="Scripted plumbing only, not model validation")
    parser.add_argument("--classify", action="store_true", help="Score early and final admission through real Clef")
    args = parser.parse_args()
    if args.fixture and args.classify:
        parser.error("--classify requires real Clef")
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
