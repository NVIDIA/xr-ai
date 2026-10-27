# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the sample's routing and control decisions over authored conversation flows.

Every turn runs through ``AgentRouter.handle()`` with the sample's destinations and
the configured Clef decision service. Answers are teacher-forced: the conversation
LLM and the observation VLM return each turn's authored completion, so the scores
measure routing, admission, native control actions, and focus, not answer quality.
Alert compilation returns a fixed watch, and alert jobs never receive a frame.

The run fails when fewer than ``_MIN_PASS_RATE`` of turns pass every check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_models import ChatResponse, ChatStreamEvent, DecisionResponse, load_models_config, make_decision
from xr_ai_routing import (
    Admission,
    AgentExchange,
    AgentRouter,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
)
from xr_ai_tools.image import ImageReference
from xr_ai_tools.vision import ImageQueryChunk
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_SAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from conversation_example_worker import NoteTakingAgent, VisualAlertAgent, VisualObservationAgent  # noqa: E402
from conversation_example_worker.app import CONVERSATION_SCOPE  # noqa: E402
from conversation_example_worker.config import load_config  # noqa: E402

_FLOWS = _SAMPLE / "eval/conversation-flows.yaml"
_CONFIG = _SAMPLE / "yaml/conversation_example_worker.yaml"
_MIN_PASS_RATE = 0.80
# Native control decisions and the destination that owns each.
_ACTION_OWNERS = {"note_action": "notes", "alert_action": "alerts"}
_WATCH = json.dumps({"status": "ready", "condition": "the requested condition is visible", "relevance": "the scene"})


@dataclass
class _Observed:
    """What one turn did: decisions, admissions, and the destination that answered."""

    route: str | None = None
    actions: dict[str, str] = field(default_factory=dict)
    admissions: dict[str, str] = field(default_factory=dict)
    answered: str | None = None
    focused: str | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)


class _Recorder:
    """Pass decisions to Clef and record their choices for the current turn."""

    def __init__(self, decisions: Any) -> None:
        self._decisions = decisions
        self.turn = _Observed()

    async def decide(self, state: Mapping[str, Any], questions: Mapping[str, Any], **kwargs: Any) -> DecisionResponse:
        response = await self._decisions.decide(state, questions, **kwargs)
        for key, answer in response.answers.items():
            self.turn.decisions.append({"question": key, "choice": answer.choice, "confidence": answer.confidence})
            if key == "destination" and self.turn.route is None:
                self.turn.route = answer.choice
            elif key in _ACTION_OWNERS:
                self.turn.actions.setdefault(_ACTION_OWNERS[key], answer.choice)
        return response


class _Watched(RoutableAgent):
    """Delegate to a sample destination, recording its admissions and answers."""

    def __init__(self, name: str, agent: RoutableAgent, recorder: _Recorder) -> None:
        self._name = name
        self._agent = agent
        self._recorder = recorder

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return self._agent.routing_aid(participant_id)

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        # The router admits a focused destination with focused=True; that is the turn's focus.
        if focused:
            self._recorder.turn.focused = self._name
        admission = await self._agent.admit(turn, focused=focused)
        self._recorder.turn.admissions[self._name] = admission.disposition
        return admission

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        self._recorder.turn.answered = self._name
        return await self._agent.answer(turn, started=started)

    async def release(self, participant_id: str) -> None:
        await self._agent.release(participant_id)

    async def participant_left(self, participant_id: str) -> None:
        await self._agent.participant_left(participant_id)


class _TeacherForced:
    """Answer conversation and observation turns with the current turn's authored completion."""

    def __init__(self) -> None:
        self.completion = ""

    async def stream_events(self, _messages: Any, **_kwargs: Any) -> AsyncIterator[ChatStreamEvent]:
        yield ChatStreamEvent(kind="text", text=self.completion)
        yield ChatStreamEvent(kind="finish", finish_reason="stop")

    async def stream(self, _request: Any) -> AsyncIterator[ImageQueryChunk]:
        yield ImageQueryChunk(text=self.completion)

    async def chat(self, _messages: Any, **_kwargs: Any) -> ChatResponse:
        # Context compaction is outside this evaluation.
        return _response("Earlier conversation.")


class _FixedWatch:
    """Compile every alert request into the same watch; compilation is not scored here."""

    async def chat(self, _messages: Any, **_kwargs: Any) -> ChatResponse:
        return _response(_WATCH)


class _Frames:
    """Give observation a placeholder frame; alert jobs wait forever for theirs."""

    def __init__(self, *, block: bool) -> None:
        self._block = block

    async def execute(self, request: Any) -> Any:
        if self._block:
            await asyncio.Event().wait()
        return SimpleNamespace(image=ImageReference(uri=f"xr-image://eval/{request.participant_id}"), timestamp_us=None)


class _NoOutput:
    async def publish(self, _topic: Any, _message: Any, **_metadata: Any) -> None:
        """Discard spoken output; only decisions and dispositions are scored."""


def _response(text: str) -> ChatResponse:
    return ChatResponse(content=text, reasoning=None, tool_calls=None, finish_reason="stop", raw={})


async def _discard(_turn: AgentTurn, _source: str, _text: str) -> None:
    pass


def load_flows(path: Path = _FLOWS) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _router(recorder: _Recorder, teacher: _TeacherForced) -> tuple[AgentRouter, VisualAlertAgent, ContextManager]:
    """Build the sample's four destinations around teacher-forced models and the recorded decisions."""

    config = load_config(_CONFIG)
    output = _NoOutput()
    context = ContextManager(teacher)
    conversation = Conversation(
        teacher,
        context=context,
        routing=RoutingAid(name="conversation", description=CONVERSATION_SCOPE),
        runtime=output,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    observation = VisualObservationAgent(_Frames(block=False), teacher, runtime=output, output_topic=VOICE_OUTPUT_TOPIC)
    alerts = VisualAlertAgent(
        _Frames(block=True),
        None,
        recorder,
        vlm=None,
        llm=_FixedWatch(),
        context=context,
        runtime=output,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    notes = NoteTakingAgent(recorder, runtime=output, output_topic=VOICE_OUTPUT_TOPIC)
    agents = {"conversation": conversation, "observation": observation, "alerts": alerts, "notes": notes}
    router = AgentRouter(
        {name: _Watched(name, agent, recorder) for name, agent in agents.items()},
        recorder,
        output=_discard,
        context_budget_bytes=config.routing_history_budget_bytes,
    )
    return router, alerts, context


def score(turn: Mapping[str, Any], observed: _Observed, result: DestinationResult | None) -> dict:
    """Return each check that ``turn`` specifies and whether ``observed`` passed it."""

    acceptable = set(turn["acceptable_destinations"])
    checks = {"destination": (observed.answered or "clarify") in acceptable}
    if observed.route is not None:
        checks["route"] = observed.route in set(turn.get("acceptable_router_choices", acceptable))
    for name, expected in turn.get("admission_expectations", {}).items():
        checks[f"admission:{name}"] = observed.admissions.get(name) == expected
    for name, expected in turn.get("destination_actions", {}).items():
        checks[f"action:{name}"] = observed.actions.get(name) == expected
    if "destination_disposition" in turn:
        checks["disposition"] = result is not None and result.disposition == turn["destination_disposition"]
    if "focus_before" in turn:
        checks["focus_before"] = observed.focused == turn["focus_before"]
    return checks


async def run(flows: list[dict[str, Any]], decisions: Any) -> list[dict[str, Any]]:
    """Run every flow through the router and return one scored record per turn."""

    recorder = _Recorder(decisions)
    teacher = _TeacherForced()
    router, alerts, context = _router(recorder, teacher)
    records = []
    try:
        for flow in flows:
            participant = f"eval:{flow['name']}"
            history = tuple(AgentExchange(row["user"], row["assistant"]) for row in flow.get("initial_history", ()))
            for index, turn in enumerate(flow["turns"]):
                recorder.turn = _Observed()
                teacher.completion = turn.get("authored_completion", "Okay.")
                # The router seeds a participant's history from its first turn only.
                request = AgentTurn(participant, f"{flow['name']}/{turn['name']}", turn["query"], history=history)
                result, error = None, None
                try:
                    result = await router.handle(request)
                except Exception as failure:  # noqa: BLE001 -- a failed turn is scored, not fatal.
                    error = f"{type(failure).__name__}: {failure}"
                checks = score(turn, recorder.turn, result)
                records.append(
                    {
                        "turn": request.turn_id,
                        "passed": error is None and all(checks.values()),
                        "checks": checks,
                        "answered": recorder.turn.answered,
                        "decisions": recorder.turn.decisions,
                        "error": error,
                    }
                )
    finally:
        await alerts.stop()
        await router.stop()
        await context.close()
    return records


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write one JSON record per turn to this file.")
    args = parser.parse_args()
    models = load_models_config(_SAMPLE / "yaml/models.json")
    decisions = make_decision(models, "router")
    try:
        records = await run(load_flows(), decisions)
    finally:
        await decisions.close()
    for record in records:
        failed = [name for name, passed in record["checks"].items() if not passed]
        detail = record["error"] or (f"failed {failed}" if failed else "")
        print(f"{'PASS' if record['passed'] else 'MISS'} {record['turn']} {detail}".rstrip())
    if args.output is not None:
        args.output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    passed = sum(record["passed"] for record in records)
    print(f"RESULT {passed}/{len(records)} turns passed")
    if passed / len(records) < _MIN_PASS_RATE:
        raise SystemExit(f"pass rate {passed / len(records):.1%} is below {_MIN_PASS_RATE:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
