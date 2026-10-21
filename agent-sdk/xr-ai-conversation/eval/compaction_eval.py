# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model eval of rolling context compaction over one long session.

Hard checks: no prompt-time drops, every committed summary within its limit,
no failed compaction, and every probed fact still present in the final prompt.
Recall: sampled probe answers must meet ``min_recall`` from ``cases/session.yaml``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from _common import (
    LogCounter,
    answer_prompt,
    answer_without_recording,
    load_cases,
    make_eval_llm,
    matches,
    normalize,
    parser,
    settle,
)
from xr_ai_conversation import Conversation, ConversationContextUpdate


def _text(spec: str | dict[str, Any]) -> str:
    """Expand a literal or a generated ``prefix``/``repeat``/``suffix`` text."""

    if isinstance(spec, str):
        return spec
    lines = [spec["repeat"].format(index=index, digit=index % 10) for index in range(spec["count"])]
    return "\n".join(filter(None, (spec.get("prefix"), *lines, spec.get("suffix"))))


async def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    session = load_cases("session.yaml")
    participant, source, budgets = session["participant"], session["source"], session["budgets"]
    samples = args.samples or session["samples"]
    llm = make_eval_llm(args)
    conversation = Conversation(
        llm,
        incremental_context_topics=(source,),
        dialogue_budget_bytes=budgets["dialogue_bytes"],
        topic_budget_bytes=budgets["topic_bytes"],
        compaction_input_budget_bytes=budgets["compaction_input_bytes"],
    )
    failures: list[str] = []
    try:
        with LogCounter() as logs:
            for index, turn in enumerate(session["turns"]):
                if "source_update" in turn:
                    await conversation._update_context(
                        participant,
                        source,
                        ConversationContextUpdate(text=_text(turn["source_update"])),
                        mode="incremental",
                        timestamp_us=time.time_ns() // 1_000,
                    )
                started = time.perf_counter()
                async for _ in conversation.stream(_text(turn["user"]), participant):
                    pass
                answered = time.perf_counter() - started
                await settle(conversation, participant)
                summary, recent = conversation._context.dialogue_snapshot(participant)
                print(
                    f"turn {index:2d}: answer {answered:4.1f}s | dialogue summary {len(summary.encode()):4d} B, "
                    f"{len(recent)} verbatim | source sections {len(conversation._context.sections(participant))}"
                )
                limit = conversation._compaction_limits(None)["summary_budget_bytes"]
                if len(summary.encode("utf-8")) > limit:
                    failures.append(f"turn {index}: dialogue summary {len(summary.encode())} B > {limit} B")
            drops = logs.count("dropping old conversation context")
            failed = logs.count("conversation context compaction failed")
            condensed = logs.count("condensing oversized context summary")
        print(f"\nprompt-time drops: {drops} | failed compactions: {failed} | condensed summaries: {condensed}")
        if drops:
            failures.append(f"{drops} prompt-time context drops")
        if failed:
            failures.append(f"{failed} failed compactions")

        summary, _ = conversation._context.dialogue_snapshot(participant)
        print(f"\ndialogue summary:\n{summary}\n")
        prompt = normalize(answer_prompt(conversation, participant, "")[1].content)
        correct = 0
        for probe in session["probes"]:
            in_prompt = any(normalize(item) in prompt for item in probe["any_of"])
            answers = await asyncio.gather(
                *(answer_without_recording(llm, conversation, participant, probe["question"]) for _ in range(samples))
            )
            passed = sum(matches(answer, probe) for answer in answers)
            correct += passed
            print(
                f"{'PASS' if in_prompt and passed * 2 > samples else 'FAIL'} {probe['name']}: "
                f"in prompt={in_prompt} answers {passed}/{samples} | {normalize(answers[0])[:90]}"
            )
            if not in_prompt:
                failures.append(f"{probe['name']}: fact missing from the final prompt context")
        recall = correct / (samples * len(session["probes"]))
        print(f"\nrecall {recall:.0%} (minimum {session['min_recall']:.0%})")
        if recall < session["min_recall"]:
            failures.append(f"recall {recall:.0%} below {session['min_recall']:.0%}")
    finally:
        await conversation.close()
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
