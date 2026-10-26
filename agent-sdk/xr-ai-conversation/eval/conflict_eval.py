# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model eval of answers when a summary disagrees with material from a different time.

Each case in ``cases/conflicts.yaml`` is seeded through the public flow with
scripted answers and summaries, then answered by the model under evaluation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

from _common import (
    ScriptedLLM,
    ask,
    load_cases,
    make_eval_llm,
    matches,
    normalize,
    parser,
    settle,
    standalone_started,
    validate_suite,
)
from xr_ai_conversation import ContextManager, Conversation, ConversationContextUpdate
from xr_ai_interaction import ResponseOutput
from xr_ai_routing import AgentTurn
from xr_ai_runtime import AgentRuntime, Topic

_PARTICIPANT = "alice"
_MINUTE_NS = 60_000_000_000


class _Clock:
    """Wall clock for one case, so seeding steps are minutes apart as in a real session."""

    def __init__(self) -> None:
        self.now_ns = time.time_ns()

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, minutes: float) -> None:
        self.now_ns += int(minutes * _MINUTE_NS)


async def _seed(
    llm: ScriptedLLM, case: dict[str, Any], clock: _Clock, step_minutes: float
) -> tuple[ContextManager, Conversation, AgentRuntime]:
    summary_topics = case.get("summary_topics", {})
    incremental_topics = case.get("incremental_topics", {})
    runtime = AgentRuntime()
    manager = ContextManager(
        llm,
        summary_context_topics=tuple(summary_topics),
        incremental_context_topics=tuple(incremental_topics),
        dialogue_budget_bytes=case.get("dialogue_budget_bytes", 8_000),
        topic_budget_bytes={**summary_topics, **incremental_topics},
    )
    conversation = Conversation(
        llm,
        context=manager,
        runtime=runtime,
        output_topic=Topic("conversation-eval.output", ResponseOutput),
    )
    if summary_topics or incremental_topics:
        runtime.register("conversation-context", manager.subscriber())
    llm.summaries = dict(case.get("summaries", {}))
    await runtime.start()
    for index, step in enumerate(case["steps"]):
        clock.advance(step_minutes)
        if "say" in step:
            llm.answer = step["say"]["assistant"]
            await conversation.answer(
                AgentTurn(_PARTICIPANT, str(index), step["say"]["user"]),
                started=standalone_started,
            )
        else:
            publication = step["publish"]
            await runtime.publish(
                Topic(publication["topic"], ConversationContextUpdate),
                ConversationContextUpdate(text=publication["text"]),
                participant_id=_PARTICIPANT,
            )
        await settle(manager, _PARTICIPANT)
    llm.answer = None
    llm.summaries = {}
    clock.advance(step_minutes)
    return manager, conversation, runtime


def _seeding_errors(manager: ContextManager, expect: dict[str, Any]) -> list[str]:
    """Check the seeded context has the shape the case is about."""

    summary, exchanges = manager._context.dialogue_snapshot(_PARTICIPANT)
    summarized = {"dialogue"} if summary.text else set()
    verbatim = [f"{entry.user}\n{entry.assistant}" for entry in exchanges]
    summary_topics = {topic.name for topic in manager._summary_context_topics}
    for source, background, records in manager._context.source_snapshots(_PARTICIPANT):
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
            clock = _Clock()
            # The runtime and the engine read wall time through time.time_ns().
            with patch("time.time_ns", clock):
                manager, conversation, runtime = await _seed(scripted, case, clock, suite["step_minutes"])
            try:
                if errors := _seeding_errors(manager, case.get("expect", {})):
                    failures.append(f"{case['name']}: seeding {'; '.join(errors)}")
                    print(f"FAIL {case['name']}: seeding {'; '.join(errors)}")
                    continue
                with patch("time.time_ns", clock):
                    answers = await asyncio.gather(
                        *(ask(llm, conversation, _PARTICIPANT, case["question"]) for _ in range(samples))
                    )
            finally:
                await conversation.close()
                await manager.close()
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
