# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from langgraph.prebuilt.tool_node import ToolInvocationError
from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import ValidationError
from tea_making_worker.applications.compose import _handle_application_request
from tea_making_worker.applications.events import APPLICATION_REQUEST, ApplicationRequest
from tea_making_worker.applications.manager.registry import ApplicationManager
from tea_making_worker.applications.manager.runtime import ApplicationOwnership
from tea_making_worker.applications.manager.spec import (
    ApplicationCatalog,
    ApplicationDescriptor,
    load_application_catalog,
)
from tea_making_worker.applications.manager.turn import ApplicationTurn, add_application_turn
from tea_making_worker.applications.manager.types import InvocationEffect, RoutedFunction
from tea_making_worker.functions.workflow import CommitRequest
from tea_making_worker.runtime.scope import current_invocation, invocation_scope
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow

_BASE = Path(__file__).parents[2] / "yaml"


class _Agent:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[str] = []

    async def ainvoke(self, request, *, to_type):
        self.requests.append(request)
        return self.answer


class _InvalidAgent(_Agent):
    async def ainvoke(self, request, *, to_type):
        self.requests.append(request)
        if len(self.requests) == 1:
            try:
                CommitRequest.model_validate({"state": "{}"})
            except ValidationError as source:
                raise ToolInvocationError("workflow__commit", source, {"state": "{}"}) from source
        return self.answer


class _RequestManager:
    function: "_RequestManager"

    def __init__(self) -> None:
        self.function = self
        self.requests: list[ApplicationTurn] = []
        self.trace_ids: list[str] = []

    async def ainvoke(self, request: ApplicationTurn, *, to_type) -> str:
        self.requests.append(request)
        self.trace_ids.append(current_invocation().trace_id)
        return "manager answer"


class _RequestOutput:
    def __init__(self) -> None:
        self.calls = []

    async def publish(self, participant_id, producer, output, **kwargs) -> str:
        self.calls.append((participant_id, producer, output, kwargs, current_invocation().trace_id))
        return output.text


class ApplicationManagerTest(unittest.IsolatedAsyncioTestCase):
    async def test_application_request_subscriber_owns_scope_and_reply_delivery(self) -> None:
        store = SessionStore(load_workflow(_BASE / "workflow.yaml"))
        manager = _RequestManager()
        output = _RequestOutput()
        event = APPLICATION_REQUEST.envelope(
            participant_id="tester",
            producer="voice.input",
            payload=ApplicationRequest(text="start making tea"),
            correlation_id="request-trace",
            timestamp_us=42,
        )

        result = await _handle_application_request(  # type: ignore[arg-type]
            manager,
            output,  # type: ignore[arg-type]
            store,
            event,
        )

        self.assertEqual(result, "manager answer")
        self.assertEqual(manager.requests, [ApplicationTurn(request="start making tea")])
        self.assertEqual(manager.trace_ids, ["request-trace"])
        self.assertEqual(output.calls[0][0:2], ("tester", "application.manager"))
        self.assertEqual(output.calls[0][3]["correlation_id"], "request-trace")
        self.assertEqual(output.calls[0][3]["parent_event_id"], event.event_id)
        self.assertEqual(output.calls[0][4], "request-trace")

    def test_ownership_supports_nested_foregrounds_and_parallel_backgrounds(self) -> None:
        spec = ApplicationCatalog(
            root_prompt="root",
            capabilities={},
            applications={
                "tea": ApplicationDescriptor("tea", "Tea", "foreground", "make tea"),
                "notes": ApplicationDescriptor("notes", "Notes", "foreground", "edit notes"),
                "watch": ApplicationDescriptor("watch", "Watch", "background", "watch scene"),
            },
        )
        ownership = ApplicationOwnership(spec)
        session = SessionStore(load_workflow(_BASE / "workflow.yaml")).get("tester")

        ownership.start_background(session, "watch")
        ownership.capture(session, "tea")
        ownership.capture(session, "notes")

        self.assertEqual(ownership.current(session), "notes")
        self.assertEqual(session.applications.background, {"watch"})
        ownership.release(session, "notes")
        self.assertEqual(ownership.current(session), "tea")
        ownership.release(session, "tea")
        self.assertEqual(ownership.current(session), "root")

    async def test_dispatch_invokes_only_the_selected_foreground(self) -> None:
        workflow = load_workflow(_BASE / "workflow.yaml")
        session = SessionStore(workflow).get("tester")
        spec = load_application_catalog(_BASE / "applications.yaml")
        ownership = ApplicationOwnership(spec)
        manager = ApplicationManager(spec, ownership)
        root_requests: list[str] = []
        tea_requests: list[str] = []

        async def root_handler(_session, request, _trace_id):
            root_requests.append(request)
            return "root answer"

        async def tea_handler(_session, request, _trace_id):
            tea_requests.append(request)
            return "tea answer"

        async with WorkflowBuilder() as builder:
            manager._root = await add_application_turn(
                builder,
                name="test__root_turn",
                description="root",
                handler=root_handler,
            )
            tea = await add_application_turn(
                builder,
                name="test__tea_turn",
                description="tea",
                handler=tea_handler,
            )
            manager.register_foreground("tea", tea)
            manager._turn = await add_application_turn(
                builder,
                name="test__manager_turn",
                description="manager",
                handler=manager._dispatch,
            )

            with invocation_scope(session, "root-trace"):
                self.assertEqual(
                    await manager.function.ainvoke(ApplicationTurn(request="hello"), to_type=str),
                    "root answer",
                )
            ownership.capture(session, "tea")
            with invocation_scope(session, "tea-trace"):
                self.assertEqual(
                    await manager.function.ainvoke(ApplicationTurn(request="next"), to_type=str),
                    "tea answer",
                )

        self.assertEqual(root_requests, ["hello"])
        self.assertEqual(tea_requests, ["next"])

    async def test_root_retries_invalid_tool_arguments_once(self) -> None:
        workflow = load_workflow(_BASE / "workflow.yaml")
        session = SessionStore(workflow).get("tester")
        spec = load_application_catalog(_BASE / "applications.yaml")
        manager = ApplicationManager(spec, ApplicationOwnership(spec))
        root = _InvalidAgent("recovered")
        manager._root_agent = root
        async with WorkflowBuilder() as builder:
            manager._root = await add_application_turn(
                builder,
                name="test__retry_root",
                description="root",
                handler=manager._route_root,
            )
            manager._turn = await add_application_turn(
                builder,
                name="test__retry_manager",
                description="manager",
                handler=manager._dispatch,
            )

            with invocation_scope(session, "trace"):
                result = await manager.function.ainvoke(
                    ApplicationTurn(request="start something"),
                    to_type=str,
                )

        self.assertEqual(result, "recovered")
        self.assertEqual(len(root.requests), 2)
        self.assertIn("state: Extra inputs are not permitted", root.requests[1])

    def test_routed_function_exposes_route_and_capture_effect(self) -> None:
        function = RoutedFunction(
            "workflow__start",
            "launch tea guidance",
            InvocationEffect.FOREGROUND,
            return_direct=True,
        )

        self.assertEqual(function.ref, "workflow__start")
        self.assertEqual(
            function.catalog_entry(),
            "workflow__start[foreground]=launch tea guidance",
        )


if __name__ == "__main__":
    unittest.main()
