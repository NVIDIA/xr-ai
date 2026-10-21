# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model eval of answers when a summary is older than material contradicting it.

Each case in ``cases/conflicts.yaml`` seeds the engine's context directly and
samples answers from the prompt the engine renders.
"""

from __future__ import annotations

import asyncio
from typing import Any

from _common import answer_without_recording, load_cases, make_eval_llm, matches, normalize, parser
from xr_ai_conversation import Conversation
from xr_ai_conversation._context import _BackgroundSection, _ContextRecord

_START_US = 1_790_856_000_000_000
_MINUTE_US = 60_000_000


def _seeded(llm: Any, case: dict[str, Any]) -> Conversation:
    summary_topics = tuple(item["source"] for item in case.get("summary_topics", ()))
    incremental = {*case.get("source_summaries", {}), *(item["source"] for item in case.get("source_updates", ()))}
    conversation = Conversation(
        llm, summary_context_topics=summary_topics, incremental_context_topics=tuple(incremental)
    )
    context = conversation._context
    for exchange in case.get("exchanges", ()):
        context.append(
            "alice",
            exchange["user"],
            exchange["assistant"],
            timestamp_us=_START_US + exchange["minute"] * _MINUTE_US,
        )
    memory = context._memory("alice")
    if "dialogue_summary" in case:
        memory.dialogue_background = _BackgroundSection(case["dialogue_summary"], -1, _START_US, _START_US, False)
    for source, text in case.get("source_summaries", {}).items():
        memory.backgrounds[source] = _BackgroundSection(text, -1, _START_US, _START_US, False)
        memory.context_order.append(source)
    for index, update in enumerate(case.get("source_updates", ())):
        if update["source"] not in memory.context_order:
            memory.context_order.append(update["source"])
        memory.context_records.setdefault(update["source"], []).append(
            _ContextRecord(index, update["text"], _START_US + update["minute"] * _MINUTE_US)
        )
    for item in case.get("summary_topics", ()):
        published = _START_US + item["minute"] * _MINUTE_US
        memory.backgrounds[item["source"]] = _BackgroundSection(item["text"], -1, published, published, False)
        memory.context_order.append(item["source"])
    return conversation


async def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    suite = load_cases("conflicts.yaml")
    samples = args.samples or suite["samples"]
    llm = make_eval_llm(args)
    failures: list[str] = []
    try:
        for case in suite["cases"]:
            conversation = _seeded(llm, case)
            answers = await asyncio.gather(
                *(answer_without_recording(llm, conversation, "alice", case["question"]) for _ in range(samples))
            )
            passed = sum(matches(answer, case) for answer in answers)
            ok = passed >= suite["min_pass_rate"] * samples
            print(f"{'PASS' if ok else 'FAIL'} {case['name']}: {passed}/{samples} | {normalize(answers[0])[:100]}")
            if not ok:
                failures.append(f"{case['name']}: {passed}/{samples} correct")
            await conversation.close()
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
