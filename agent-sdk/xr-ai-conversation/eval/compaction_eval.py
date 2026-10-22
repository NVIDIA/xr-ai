# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-model eval of rolling context compaction over long sessions.

Each session runs through the engine's public flow with the model under
evaluation. A session fails on a failed compaction, on context omitted from a
prompt, when a probe's ``in_prompt`` phrases (default ``any_of``) are missing
from the final prompt context, or when a probe is answered correctly in fewer
than half of its samples.
"""

from __future__ import annotations

import asyncio
import time

from _common import (
    LogCounter,
    ask,
    expand,
    load_cases,
    make_eval_llm,
    matches,
    normalize,
    parser,
    settle,
    standalone_started,
    validate_suite,
)
from xr_ai_conversation import Conversation, ConversationContextUpdate
from xr_ai_models import LLMService
from xr_ai_routing import AgentTurn
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_voice import VoiceOutput

SESSIONS = ("session_lab.yaml", "session_renovation.yaml")


async def run_session(llm: LLMService, name: str, sample_override: int | None) -> list[str]:
    session = load_cases(name)
    validate_suite(session)
    participant, topic_name, budgets = session["participant"], session["topic"], session["budgets"]
    samples = sample_override or session["samples"]
    topic = Topic(topic_name, ConversationContextUpdate)
    runtime = AgentRuntime()
    conversation = Conversation(
        llm,
        runtime=runtime,
        output_topic=Topic("conversation-eval.output", VoiceOutput),
        incremental_context_topics=(topic_name,),
        dialogue_budget_bytes=budgets["dialogue_bytes"],
        topic_budget_bytes=budgets["topic_bytes"],
    )
    runtime.register("conversation-context", conversation.context_subscriber())
    failures: list[str] = []
    print(f"=== {name}")
    try:
        async with runtime:
            with LogCounter() as logs:
                for index, turn in enumerate(session["turns"]):
                    if "topic_update" in turn:
                        await runtime.publish(
                            topic,
                            ConversationContextUpdate(text=expand(turn["topic_update"])),
                            participant_id=participant,
                        )
                    started = time.perf_counter()
                    await conversation.answer(
                        AgentTurn(participant, str(index), expand(turn["user"])), started=standalone_started,
                    )
                    answered = time.perf_counter() - started
                    await settle(conversation, participant)
                    summary, recent = conversation._context.dialogue_snapshot(participant)
                    print(
                        f"turn {index:2d}: answer {answered:4.1f}s | dialogue summary "
                        f"{len(summary.text.encode()):4d} B, {len(recent)} verbatim"
                    )
                failed = logs.count("conversation context compaction failed")
                omitted = logs.count("omitting uncompacted context")
            print(f"failed compactions: {failed} | prompt omissions: {omitted}")
            if failed:
                failures.append(f"{name}: {failed} failed compactions")
            if omitted:
                failures.append(f"{name}: {omitted} prompt omissions")

            context = normalize(conversation._answer_messages(participant, "")[1].content)
            for probe in session["probes"]:
                in_prompt = matches(context, {"any_of": probe.get("in_prompt", probe["any_of"])})
                answers = await asyncio.gather(
                    *(ask(llm, conversation, participant, probe["question"]) for _ in range(samples))
                )
                passed = sum(matches(answer, probe) for answer in answers)
                ok = in_prompt and passed * 2 > samples
                print(
                    f"{'PASS' if ok else 'FAIL'} {probe['name']}: in prompt={in_prompt} answers {passed}/{samples} "
                    f"| {normalize(answers[0])[:80]}"
                )
                if not in_prompt:
                    failures.append(f"{name} {probe['name']}: fact missing from the final prompt context")
                elif not ok:
                    failures.append(f"{name} {probe['name']}: {passed}/{samples} correct")
    finally:
        await conversation.close()
    return failures


async def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    llm = make_eval_llm(args)
    failures: list[str] = []
    try:
        for name in SESSIONS:
            failures += await run_session(llm, name, args.samples)
    finally:
        await llm.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
