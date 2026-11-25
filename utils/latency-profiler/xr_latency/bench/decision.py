# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision-service (router model) latency against state size and label count."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from xr_ai_models import DecisionQuestion, DecisionService

from .._gpu import load_tags
from .._store import RunStore
from .._timing import now_ms

HISTORY_BYTES = (0, 1_000, 3_000, 6_000, 12_000)
LABEL_COUNTS = (2, 4, 8)


@dataclass
class DecisionPlan:
    """Sample count and sweeps for the decision benchmark."""

    samples: int = 30
    history_bytes: tuple[int, ...] = HISTORY_BYTES
    label_counts: tuple[int, ...] = LABEL_COUNTS
    concurrency: tuple[int, ...] = (2, 4)


def _history(nbytes: int) -> list[dict[str, str]]:
    """Whole exchanges totalling about ``nbytes`` of JSON-ish text."""
    exchanges, size, i = [], 0, 0
    while size < nbytes:
        item = {
            "user": f"Can you check gauge {i} and tell me whether the pressure reading looks normal today?",
            "assistant": f"Gauge {i} reads normal. The temperature drift on the second panel is still within limits.",
            "recorded_around_utc": "2026-10-22 17:41",
        }
        exchanges.append(item)
        size += sum(len(v) for v in item.values()) + 40
        i += 1
    return exchanges


def question(labels: int) -> dict[str, DecisionQuestion]:
    """A router-shaped question with ``labels`` destinations plus ``clarify``."""
    names = [f"destination_{i}" for i in range(labels)]
    criteria = {n: f"{n} handles requests about topic {i}; eligible=True; priority=0." for i, n in enumerate(names)}
    criteria["clarify"] = "The request cannot safely be assigned to one available destination."
    return {"destination": DecisionQuestion(
        instructions="Choose the best eligible destination using its description and routing hints.",
        criteria=criteria,
    )}


async def run(decision: DecisionService, store: RunStore, plan: DecisionPlan) -> None:
    """Sweep history size and label count, recording latency, chosen label and confidence."""
    for labels in plan.label_counts:
        for nbytes in plan.history_bytes:
            questions = question(labels)
            for i in range(plan.samples):
                state = {
                    "participant_id": "p1",
                    "request": f"Please look at topic {i % labels} for me.",
                    "completed_history": _history(nbytes),
                    "destinations": [{"name": f"destination_{j}", "description": f"topic {j}"} for j in range(labels)],
                }
                start = now_ms()
                response = await decision.decide(state, questions)
                total = now_ms() - start
                answer = response.answers["destination"]
                store.add(
                    "decision", {"labels": labels, "history_bytes": nbytes, **load_tags()},
                    total=total, choice=answer.choice, confidence=answer.confidence,
                    expected=f"destination_{i % labels}",
                    prompt_tokens=(response.usage or {}).get("prompt_tokens"),
                )
    await _concurrency(decision, store, plan)


async def _concurrency(decision: DecisionService, store: RunStore, plan: DecisionPlan) -> None:
    """N simultaneous decides: the server has one worker, so the router's route + slow-prediction calls may queue."""
    questions = question(4)
    state = {
        "participant_id": "p1", "request": "Please look at topic 1 for me.",
        "completed_history": _history(3_000),
        "destinations": [{"name": f"destination_{j}", "description": f"topic {j}"} for j in range(4)],
    }
    for width in plan.concurrency:
        for _ in range(max(plan.samples // 3, 5)):
            async def one() -> float:
                start = now_ms()
                await decision.decide(state, questions)
                return now_ms() - start

            times = await asyncio.gather(*(one() for _ in range(width)))
            for rank, total in enumerate(sorted(times)):
                store.add("decision.concurrency", {"width": width, "rank": rank, **load_tags()},
                          total=total)
