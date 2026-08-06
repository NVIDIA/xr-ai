# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path

from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import ValidationError
from tea_making_worker.agents.registry import AgentRegistry, _state_contract
from tea_making_worker.functions.workflow import CommitRequest
from tea_making_worker.runtime.scope import current_invocation, invocation_scope
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


class _Capture:
    request = ""

    async def ainvoke(self, request, *, to_type):
        self.request = request
        return "captured"


class _InvalidCommit:
    def __init__(self, *, always: bool = False) -> None:
        self.always = always
        self.requests: list[str] = []

    async def ainvoke(self, request, *, to_type):
        self.requests.append(request)
        if self.always or len(self.requests) == 1:
            try:
                CommitRequest.model_validate({"state": "{}"})
            except ValidationError as source:
                raise ToolInvocationError(
                    "workflow__commit",
                    source,
                    {"state": "{}"},
                ) from source
        return "recovered"


class _Broken:
    async def ainvoke(self, request, *, to_type):
        raise RuntimeError("service failed")


class _RouteAgent:
    def __init__(self, *, select_on: int | None) -> None:
        self.select_on = select_on
        self.requests: list[str] = []

    async def ainvoke(self, request, *, to_type):
        self.requests.append(request)
        if self.select_on == len(self.requests):
            current_invocation().route_operation = "ask_general"
            return "general answer"
        return "unrouted answer"


class RegistryTest(unittest.IsolatedAsyncioTestCase):
    async def test_observation_context_explains_state_contract(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        session = store.get("tester")
        store.start(session)
        capture = _Capture()
        registry = AgentRegistry(workflow)
        registry._step["identify"] = capture

        await registry.observe(session, "visible label", "trace")

        self.assertEqual(
            json.loads(capture.request),
            {
                "observation": "visible label",
                "already_complete": False,
                "state": {},
            },
        )
        contract = _state_contract(workflow, workflow.step("identify"))
        self.assertIn("Writable state:", contract)
        self.assertIn("tea_name:string — Tea name from the current caption", contract)
        self.assertIn("tea_ready:boolean — True only when all preceding fields", contract)
        self.assertIn("Completion requires tea_ready=true", contract)

        session.state["tea_ready"] = True
        await registry.observe(session, "another frame", "complete-trace")
        self.assertTrue(json.loads(capture.request)["already_complete"])

    async def test_observation_retries_invalid_tool_arguments_once(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("tester")
        session.active = True
        session.step_id = "identify"
        agent = _InvalidCommit()
        registry = AgentRegistry(workflow)
        registry._step["identify"] = agent

        result = await registry.observe(session, "visible label", "trace")

        self.assertEqual(result, "recovered")
        self.assertEqual(len(agent.requests), 2)
        self.assertIn("state: Extra inputs are not permitted", agent.requests[1])

    async def test_repeated_invalid_tool_arguments_skip_only_the_frame(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("tester")
        session.active = True
        session.step_id = "identify"
        agent = _InvalidCommit(always=True)
        registry = AgentRegistry(workflow)
        registry._step["identify"] = agent

        self.assertEqual(await registry.observe(session, "visible label", "trace"), "")
        self.assertEqual(len(agent.requests), 2)

    async def test_non_schema_agent_errors_propagate(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("tester")
        session.active = True
        session.step_id = "identify"
        registry = AgentRegistry(workflow)
        registry._step["identify"] = _Broken()

        with self.assertRaisesRegex(RuntimeError, "service failed"):
            await registry.observe(session, "visible label", "trace")

    async def test_router_retries_a_direct_answer_until_a_tool_is_selected(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("tester")
        agent = _RouteAgent(select_on=2)
        registry = AgentRegistry(workflow)
        registry._router = agent

        with invocation_scope(session, "trace"):
            result = await registry.route(session, "What tea is this?", "trace")

        self.assertEqual(result, "general answer")
        self.assertEqual(len(agent.requests), 2)
        self.assertIn("Call exactly one route tool.", agent.requests[1])

    async def test_router_returns_a_recoverable_reply_after_two_direct_answers(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("tester")
        agent = _RouteAgent(select_on=None)
        registry = AgentRegistry(workflow)
        registry._router = agent

        with invocation_scope(session, "trace"):
            result = await registry.route(session, "ambiguous request", "trace")

        self.assertEqual(result, "I could not determine the request. Please rephrase it.")
        self.assertEqual(len(agent.requests), 2)


if __name__ == "__main__":
    unittest.main()
