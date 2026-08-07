# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path

from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import ValidationError
from tea_making_worker.desktop.registry import Desktop
from tea_making_worker.desktop.runtime import DesktopRuntime
from tea_making_worker.desktop.spec import ApplicationSpec, DesktopSpec, load_desktop
from tea_making_worker.desktop.types import FunctionEffect, RoutedFunction
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


class _Foreground:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[str] = []

    async def route(self, session, request, trace_id) -> str:
        self.requests.append(request)
        return self.answer


class DesktopTest(unittest.IsolatedAsyncioTestCase):
    def test_runtime_supports_nested_foregrounds_and_parallel_backgrounds(self) -> None:
        spec = DesktopSpec(
            root_prompt="root",
            capabilities={},
            applications={
                "tea": ApplicationSpec("tea", "Tea", "foreground", "make tea"),
                "notes": ApplicationSpec("notes", "Notes", "foreground", "edit notes"),
                "watch": ApplicationSpec("watch", "Watch", "background", "watch scene"),
            },
        )
        runtime = DesktopRuntime(spec)
        session = SessionStore(load_workflow(_BASE / "workflow.yaml")).get("tester")

        runtime.start_background(session, "watch")
        runtime.capture(session, "tea")
        runtime.capture(session, "notes")

        self.assertEqual(runtime.current(session), "notes")
        self.assertEqual(session.desktop.background, {"watch"})
        runtime.release(session, "notes")
        self.assertEqual(runtime.current(session), "tea")
        runtime.release(session, "tea")
        self.assertEqual(runtime.current(session), "root")

    async def test_dispatch_invokes_only_the_selected_foreground(self) -> None:
        workflow = load_workflow(_BASE / "workflow.yaml")
        session = SessionStore(workflow).get("tester")
        spec = load_desktop(_BASE / "applications.yaml")
        runtime = DesktopRuntime(spec)
        desktop = Desktop(spec, runtime)
        root = _Agent("root answer")
        tea = _Foreground("tea answer")
        desktop._root = root
        desktop.register_foreground("tea", tea)

        with invocation_scope(session, "root-trace"):
            self.assertEqual(await desktop.route(session, "hello", "root-trace"), "root answer")
        runtime.capture(session, "tea")
        with invocation_scope(session, "tea-trace"):
            self.assertEqual(await desktop.route(session, "next", "tea-trace"), "tea answer")

        self.assertEqual(json.loads(root.requests[0]), {"request": "hello"})
        self.assertEqual(tea.requests, ["next"])
        self.assertEqual(len(root.requests), 1)

    async def test_root_retries_invalid_tool_arguments_once(self) -> None:
        workflow = load_workflow(_BASE / "workflow.yaml")
        session = SessionStore(workflow).get("tester")
        spec = load_desktop(_BASE / "applications.yaml")
        desktop = Desktop(spec, DesktopRuntime(spec))
        root = _InvalidAgent("recovered")
        desktop._root = root

        with invocation_scope(session, "trace"):
            result = await desktop.route(session, "start something", "trace")

        self.assertEqual(result, "recovered")
        self.assertEqual(len(root.requests), 2)
        self.assertIn("state: Extra inputs are not permitted", root.requests[1])

    def test_routed_function_exposes_route_and_capture_effect(self) -> None:
        function = RoutedFunction(
            "workflow__start",
            "launch tea guidance",
            FunctionEffect.FOREGROUND,
            return_direct=True,
        )

        self.assertEqual(function.ref, "workflow__start")
        self.assertEqual(
            function.catalog_entry(),
            "workflow__start[foreground]=launch tea guidance",
        )


if __name__ == "__main__":
    unittest.main()
