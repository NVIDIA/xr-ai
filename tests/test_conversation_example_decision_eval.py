# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decision eval runs every flow through the router and scores what actually happened."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from xr_ai_models import DecisionAnswer, DecisionResponse

_RUNNER = Path(__file__).resolve().parents[1] / "agent-samples/conversation-example/eval/run_decision_eval.py"
_SPEC = importlib.util.spec_from_file_location("conversation_example_decision_eval", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)

_DEFAULT_ACTIONS = {"note_action": ("handoff", "other"), "alert_action": ("decline",)}


class _FixtureDecisions:
    """Answer each decision as its turn's fixture expects; a plumbing fake, not a model result."""

    def __init__(self, flows, *, route=None):
        self.turns = {turn["query"]: turn for flow in flows for turn in flow["turns"]}
        self.route = route

    async def decide(self, state, questions, **_kwargs):
        turn = self.turns[state["request"]]
        answers = {}
        for key, question in questions.items():
            criteria = list(question.criteria)
            if key == "destination":
                choices = [self.route] if self.route else turn.get("acceptable_router_choices", [])
                choices = [*choices, *turn["acceptable_destinations"]]
            else:
                expected = turn.get("destination_actions", {}).get(evaluator._ACTION_OWNERS[key])
                choices = [*([expected] if expected else []), *_DEFAULT_ACTIONS[key]]
            choice = next(item for item in choices if item in criteria)
            if choice == "clarify":
                # Uncertain routing probes the most probable eligible destination.
                probabilities = {item: 0.6 if item == "clarify" else 0.4 / (len(criteria) - 1) for item in criteria}
            else:
                probabilities = {item: float(item == choice) for item in criteria}
            answers[key] = DecisionAnswer(choice, probabilities[choice], probabilities)
        return DecisionResponse(answers, "fixture-double", {}, {})


@pytest.mark.asyncio
async def test_fixture_faithful_decisions_pass_every_turn():
    flows = evaluator.load_flows()
    records = await evaluator.run(flows, _FixtureDecisions(flows))

    assert len(records) == sum(len(flow["turns"]) for flow in flows)
    assert [record for record in records if not record["passed"]] == []
    checks = {name.split(":")[0] for record in records for name in record["checks"]}
    assert checks == {"destination", "route", "admission", "action", "disposition", "focus_before"}


@pytest.mark.asyncio
async def test_wrong_routes_are_scored_as_misses():
    flows = evaluator.load_flows()
    records = await evaluator.run(flows, _FixtureDecisions(flows, route="observation"))

    missed = {record["turn"] for record in records if not record["passed"]}
    assert "general_planning_continuation/start_general_plan" in missed
    assert "current_visual_then_recall/current_view_over_old_observation" not in missed
