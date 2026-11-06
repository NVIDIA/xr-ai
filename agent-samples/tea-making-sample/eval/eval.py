# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay control/Q&A cases through the real runner, SDK router and Conversation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from tea_making_worker.application.application_config import load_config
from tea_making_worker.application.tea_events import USER_QUERY_TOPIC
from tea_making_worker.application.tea_sop_agent import SOP_CONTEXT_SOURCE, SOPAgent
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_hub import FrameUnavailable
from xr_ai_models import load_models_config, make_decision, make_llm
from xr_ai_routing import AgentRouter, AgentTurn, RoutingAid
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.rag import RetrieveResult
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_SAMPLE = Path(__file__).resolve().parents[1]


class _Decisions:
    """Keep actual bounded-choice requests/results, including routing and authorization."""

    def __init__(self, service):
        self.service = service
        self.calls = []

    async def decide(self, state, questions, **kwargs):
        started = time.perf_counter()
        result = await self.service.decide(state, questions, **kwargs)
        self.calls.append(
            {
                "state": state,
                "questions": {
                    key: {"instructions": question.instructions, "criteria": dict(question.criteria)}
                    for key, question in questions.items()
                },
                "answers": {key: answer.choice for key, answer in result.answers.items()},
                "probabilities": {key: dict(answer.probabilities) for key, answer in result.answers.items()},
                "duration_ms": (time.perf_counter() - started) * 1000,
            }
        )
        return result


async def main():
    """Use production composition; omit camera input rather than fabricating visual evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_SAMPLE / "eval/cases.yaml")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    cases = yaml.safe_load(arguments.cases.read_text())
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    models = load_models_config(config.models_config)
    llm, service = make_llm(models, "llm"), make_decision(models, "decision")
    decisions = _Decisions(service)
    conversation_requests = []

    async def record_conversation_stream(messages):
        """Observe actual SDK-built messages without changing conversation behavior."""
        conversation_requests.append(messages)
        stream = llm.stream_events(messages)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    runtime = AgentRuntime()
    context = ContextManager(
        llm,
        summary_context_topics=(SOP_CONTEXT_SOURCE,),
        dialogue_budget_bytes=config.context_budget_bytes,
        topic_budget_bytes=config.sop_context_budget_bytes,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )

    async def no_camera(_request):
        raise FrameUnavailable("Routing evaluation supplies no camera image.")

    async def no_passages(_request):
        return RetrieveResult(results=[])

    async def control_output(_turn, _route, _text):
        """Destinations still publish their normal SDK outputs to this local runtime."""

    sop = SOPAgent(
        config=config.sop,
        llm=llm,
        decisions=decisions,
        frames=SimpleNamespace(execute=no_camera),
        image_query=SimpleNamespace(),
        rag=SimpleNamespace(retrieve=SimpleNamespace(execute=no_passages)),
        context=context,
        runtime=runtime,
    )
    conversation = Conversation(
        SimpleNamespace(stream_events=record_conversation_stream),
        context=context,
        system_prompt=config.conversation_system_prompt,
        instructions=config.sop.conversation_instructions,
        routing=RoutingAid("conversation", config.sop.conversation_description),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    router = AgentRouter(
        {"workflow": sop, "conversation": conversation},
        decisions,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        output=control_output,
        initialize_participant=sop.participant_joined,
        cleanup_participant=sop.participant_left,
    )
    runtime.register("routing", router.runtime_subscriber())
    rows = []
    try:
        async with runtime:
            for index, case in enumerate(cases):
                participant = f"eval-{index}"
                for setup_index, query in enumerate(case.get("setup_requests", [])):
                    await router.handle(AgentTurn(participant, f"setup-{index}-{setup_index}", query))
                first_call = len(decisions.calls)
                first_answer_request = len(conversation_requests)
                started = time.perf_counter()
                result = await router.handle(AgentTurn(participant, f"case-{index}", case["query"]))
                calls = decisions.calls[first_call:]
                destinations = [call["answers"]["destination"] for call in calls if "destination" in call["answers"]]
                controls = [call["answers"]["control"] for call in calls if "control" in call["answers"]]
                authorized = any(
                    call["answers"].get("authorization") == "command"
                    and call["probabilities"]["authorization"]["command"]
                    > sum(
                        probability
                        for label, probability in call["probabilities"]["authorization"].items()
                        if label != "command"
                    )
                    for call in calls
                )
                executed_control = controls[-1] if controls and controls[-1] != "none" and authorized else None
                destination = "conversation" if len(conversation_requests) > first_answer_request else destinations[-1]
                passed = destination == case["expected_destination"] and executed_control == case.get(
                    "expected_control"
                )
                row = {
                    "case": case["name"],
                    "query": case["query"],
                    "passed": passed,
                    "destination": destination,
                    "control": executed_control,
                    "response": result.text,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "sop": sop.context_snapshot(participant),
                    "decisions": calls,
                    "conversation_messages": [
                        {"role": message.role, "content": message.content}
                        for request in conversation_requests[first_answer_request:]
                        for message in request
                    ],
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
                await sop.participant_left(participant)
            await router.stop()
            await sop.stop()
    finally:
        await conversation.close()
        await context.close()
        await llm.close()
        await service.close()
    if arguments.output:
        arguments.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
