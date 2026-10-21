# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model eval of answers when a summary disagrees with material from a different time.

Each case in ``cases/conflicts.yaml`` is seeded through the public flow with
scripted answers and summaries, then answered by the model under evaluation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from _common import ScriptedLLM, ask, load_cases, make_eval_llm, matches, normalize, parser, settle, validate_suite
from xr_ai_conversation import Conversation, ConversationContextUpdate
from xr_ai_runtime import AgentRuntime, Topic

_PARTICIPANT = "alice"


async def _seed(llm: ScriptedLLM, case: dict[str, Any]) -> tuple[Conversation, AgentRuntime]:
    summary_topics = case.get("summary_topics", {})
    incremental_topics = case.get("incremental_topics", {})
    conversation = Conversation(
        llm,
        summary_context_topics=tuple(summary_topics),
        incremental_context_topics=tuple(incremental_topics),
        dialogue_budget_bytes=case.get("dialogue_budget_bytes", 8_000),
        topic_budget_bytes={**summary_topics, **incremental_topics},
    )
    runtime = AgentRuntime()
    if summary_topics or incremental_topics:
        runtime.register("conversation-context", conversation.context_subscriber())
    llm.summaries = dict(case.get("summaries", {}))
    await runtime.start()
    for step in case["steps"]:
        if "say" in step:
            llm.answer = step["say"]["assistant"]
            async for _ in conversation.stream(step["say"]["user"], _PARTICIPANT):
                pass
        else:
            publication = step["publish"]
            await runtime.publish(
                Topic(publication["topic"], ConversationContextUpdate),
                ConversationContextUpdate(text=publication["text"]),
                participant_id=_PARTICIPANT,
            )
        await settle(conversation, _PARTICIPANT)
    llm.answer = None
    llm.summaries = {}
    return conversation, runtime


def _seeding_errors(conversation: Conversation, expect: dict[str, Any]) -> list[str]:
    """Check the seeded context has the shape the case is about."""

    summary, exchanges = conversation._context.dialogue_snapshot(_PARTICIPANT)
    summarized = {"dialogue"} if summary.text else set()
    verbatim = [f"{entry.user}\n{entry.assistant}" for entry in exchanges]
    summary_topics = {topic.name for topic in conversation._summary_context_topics}
    for source, background, records in conversation._context.source_snapshots(_PARTICIPANT):
        if background is not None and source in summary_topics:
            verbatim.append(background.text)
        elif background is not None:
            summarized.add(source)
        verbatim.extend(record.text for record in records)
    errors = [f"{section} was not summarized" for section in expect.get("summarized", ()) if section not in summarized]
    errors += [
        f"{text!r} is not verbatim" for text in expect.get("verbatim", ()) if not any(text in v for v in verbatim)
    ]
    return errors


async def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    suite = load_cases("conflicts.yaml")
    validate_suite(suite)
    samples = args.samples or suite["samples"]
    llm = make_eval_llm(args)
    failures: list[str] = []
    try:
        for case in suite["cases"]:
            scripted = ScriptedLLM(llm)
            conversation, runtime = await _seed(scripted, case)
            try:
                if errors := _seeding_errors(conversation, case.get("expect", {})):
                    failures.append(f"{case['name']}: seeding {'; '.join(errors)}")
                    print(f"FAIL {case['name']}: seeding {'; '.join(errors)}")
                    continue
                answers = await asyncio.gather(
                    *(ask(llm, conversation, _PARTICIPANT, case["question"]) for _ in range(samples))
                )
            finally:
                await conversation.close()
                await runtime.stop()
            passed = sum(matches(answer, case) for answer in answers)
            ok = passed >= suite["min_pass_rate"] * samples
            print(f"{'PASS' if ok else 'FAIL'} {case['name']}: {passed}/{samples} | {normalize(answers[0])[:100]}")
            if not ok:
                failures.append(f"{case['name']}: {passed}/{samples} correct")
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
