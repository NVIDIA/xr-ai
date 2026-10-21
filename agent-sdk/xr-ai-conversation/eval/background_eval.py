# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model eval of background compaction cost and its effect on foreground answers.

For each ``compaction_thinking_budget`` in ``cases/background.yaml`` it reports
compaction duration, summary size and kept facts, and checks that foreground
first-token latency stays close to idle while a compaction request runs.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

from _common import load_cases, make_eval_llm, normalize, parser
from compaction_eval import _text
from xr_ai_conversation import Conversation
from xr_ai_conversation._context import _TranscriptEntry
from xr_ai_models import ChatMessage, LLMService


def _exchanges() -> tuple[_TranscriptEntry, ...]:
    session = load_cases("session.yaml")
    return tuple(
        _TranscriptEntry(index, "dialogue", _text(turn["user"]), "Noted.", None)
        for index, turn in enumerate(session["turns"])
    )


async def _compact(llm: LLMService, thinking_budget: int | None, limit: int) -> tuple[float, str]:
    """Merge the session in two requests, as rolling compaction would."""

    conversation = Conversation(llm, compaction_thinking_budget=thinking_budget)
    exchanges = _exchanges()
    half = len(exchanges) // 2
    started = time.perf_counter()
    summary = await conversation._compact_dialogue("", exchanges[:half], limit)
    summary = await conversation._compact_dialogue(f"Earlier summary:\n{summary}", exchanges[half:], limit)
    return time.perf_counter() - started, summary


async def _foreground(llm: LLMService, spec: dict[str, Any]) -> float:
    messages = (
        ChatMessage(role="system", content="Answer concisely."),
        ChatMessage(role="user", content=spec["context"]),
        ChatMessage(role="user", content=spec["question"]),
    )
    started = time.perf_counter()
    first = None
    async for event in llm.stream_events(messages):
        if event.kind == "text" and event.text and first is None:
            first = time.perf_counter() - started
    return first if first is not None else time.perf_counter() - started


async def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    suite = load_cases("background.yaml")
    trials, limit = args.samples or suite["trials"], suite["summary_limit_bytes"]
    llm = make_eval_llm(args)
    failures: list[str] = []
    try:
        print(f"=== compaction (two merge requests, summary limit {limit} B) ===")
        for budget in suite["thinking_budgets"]:
            runs = await asyncio.gather(*(_compact(llm, budget, limit) for _ in range(trials)), return_exceptions=True)
            done = [run for run in runs if not isinstance(run, BaseException)]
            sizes = [len(summary.encode("utf-8")) for _, summary in done]
            kept = {
                fact: sum(any(item in normalize(summary) for item in needles) for _, summary in done)
                for fact, needles in suite["facts"].items()
            }
            label = "model default" if budget is None else f"thinking {budget}"
            durations = [duration for duration, _ in done] or [float("nan")]
            print(
                f"{label:14s} median {statistics.median(durations):5.1f}s max {max(durations):5.1f}s | "
                f"errors {len(runs) - len(done)} | empty {sizes.count(0)} | bytes {min(sizes, default=0)}-"
                f"{max(sizes, default=0)} (over limit {sum(size > limit for size in sizes)})"
            )
            print("   kept: " + ", ".join(f"{fact} {count}/{len(done)}" for fact, count in kept.items()))

        print("\n=== foreground first-token latency while one compaction request runs ===")
        idle = None
        for budget in ("idle", *suite["thinking_budgets"]):
            firsts = []
            for _ in range(trials + 2):
                background = None
                if budget != "idle":
                    background = asyncio.create_task(_compact(llm, budget, limit))
                    await asyncio.sleep(0.3)
                firsts.append(await _foreground(llm, suite["foreground"]))
                if background is not None:
                    background.cancel()
                    await asyncio.gather(background, return_exceptions=True)
            median_ms = statistics.median(firsts) * 1000
            idle = median_ms if budget == "idle" else idle
            label = budget if budget == "idle" else ("model default" if budget is None else f"thinking {budget}")
            print(f"{label:14s} median {median_ms:6.0f} ms max {max(firsts) * 1000:6.0f} ms")
            if idle is not None and median_ms - idle > suite["max_first_token_slowdown_ms"]:
                failures.append(f"{label}: first token {median_ms - idle:.0f} ms slower than idle")
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
