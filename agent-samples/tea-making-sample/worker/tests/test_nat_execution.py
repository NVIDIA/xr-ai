# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import ValidationError
from tea_making_worker.applications.change_watch import ChangeWatchApplication
from tea_making_worker.applications.controls import add_background_controls
from tea_making_worker.desktop.runtime import DesktopRuntime
from tea_making_worker.desktop.spec import load_desktop
from tea_making_worker.functions import (
    CurrentViewConfig,
    CurrentViewRequest,
    TemperatureVerifyConfig,
    TemperatureVerifyRequest,
    add_workflow_functions,
)
from tea_making_worker.functions.temperature import temperature_verify
from tea_making_worker.functions.vision import current_view
from tea_making_worker.functions.workflow import GuideRequest
from tea_making_worker.runtime.scope import current_invocation, invocation_scope
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow
from xr_ai_nat.functions.vision import LiveVisionResult

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"
_APPLICATIONS = Path(__file__).parents[2] / "yaml" / "applications.yaml"


class NatExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_scope_excludes_timers_and_appliances(self) -> None:
        self.assertEqual(GuideRequest(scope="tea_guide").scope, "tea_guide")
        with self.assertRaises(ValidationError):
            GuideRequest.model_validate({"scope": "timer"})

    async def test_temperature_verification_converts_and_compares(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("temperature-test")
        cases = (
            (100, TemperatureVerifyRequest(reading=73, unit="fahrenheit"), False),
            (100, TemperatureVerifyRequest(reading=212, unit="fahrenheit"), True),
            (98, TemperatureVerifyRequest(reading=70, unit="celsius"), False),
            (82, TemperatureVerifyRequest(reading=100, unit="celsius"), True),
        )

        async with temperature_verify(TemperatureVerifyConfig(), None) as info:
            self.assertIsNotNone(info.single_fn)
            self.assertEqual(set(TemperatureVerifyRequest.model_fields), {"reading", "unit"})
            with invocation_scope(session, "temperature-trace"):
                for target_c, request, expected in cases:
                    session.state["target_temperature_c"] = target_c
                    state = dict(session.state)
                    result = await info.single_fn(request)
                    self.assertIs(result.ready, expected)
                    self.assertEqual(result.target_c, target_c)
                    self.assertEqual(session.state, state)

    async def test_current_view_injects_the_active_participant(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("actual-participant")

        class _Source:
            request = None

            async def ainvoke(self, request, *, to_type):
                self.request = request
                return LiveVisionResult(answer="visible answer")

        source = _Source()
        async with current_view(CurrentViewConfig(source=source), None) as info:
            self.assertIsNotNone(info.single_fn)
            with invocation_scope(session, "voice-trace"):
                result = await info.single_fn(CurrentViewRequest(question="What is visible?"))

        self.assertEqual(result.answer, "visible answer")
        self.assertEqual(source.request.participant_id, "actual-participant")
        self.assertEqual(source.request.question, "What is visible?")

    async def test_current_view_timeout_is_a_recoverable_observation(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        session = SessionStore(workflow).get("actual-participant")

        class _Source:
            async def ainvoke(self, _request, *, to_type):
                await asyncio.sleep(1)
                return to_type(answer="late answer")

        async with current_view(CurrentViewConfig(source=_Source(), timeout_s=0.01), None) as info:
            self.assertIsNotNone(info.single_fn)
            with invocation_scope(session, "timeout-trace"):
                result = await info.single_fn(CurrentViewRequest(question="What is visible?"))

        self.assertIn("Unable to inspect", result.answer)

    async def test_workflow_nat_functions_are_the_only_mutation_surface(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        desktop = DesktopRuntime(load_desktop(_APPLICATIONS))
        store = SessionStore(workflow)
        session = store.get("tester")

        async with WorkflowBuilder() as builder:
            await add_workflow_functions(builder, store=store, desktop=desktop)
            start = await builder.get_function("workflow__start")
            restart = await builder.get_function("workflow__restart")
            commit = await builder.get_function("workflow__commit")
            with invocation_scope(session, "route-trace"):
                response = await start.ainvoke({"scope": "tea_guide"}, to_type=str)
                self.assertEqual(current_invocation().route_operation, "start")
            self.assertEqual(response, workflow.step("identify").enter_message)
            self.assertEqual(session.step_id, "identify")
            self.assertEqual(desktop.current(session), "tea")
            session.step_id = "heat_water"
            with invocation_scope(session, "repeated-start-trace"):
                response = await start.ainvoke({"scope": "tea_guide"}, to_type=str)
            self.assertEqual(response, "Current step: Heat the water.")
            self.assertEqual(session.step_id, "heat_water")
            with invocation_scope(session, "restart-trace"):
                response = await restart.ainvoke({"scope": "tea_guide"}, to_type=str)
                self.assertEqual(current_invocation().route_operation, "restart")
            self.assertEqual(response, workflow.step("identify").enter_message)
            self.assertEqual(session.step_id, "identify")
            store.observe(
                session,
                "A tea package label reads Oolong, 88 C, steep 4 minutes.",
                "identify",
            )

            with invocation_scope(session, "step-trace"):
                response = await commit.ainvoke(
                    {
                        "updates": {
                            "tea_name": "oolong",
                            "target_temperature_c": 88,
                            "steep_duration_s": 240,
                            "guidance_source": "package",
                            "tea_ready": True,
                        },
                        "message": "",
                    },
                    to_type=str,
                )
            self.assertIn('"accepted":true', response)
            self.assertEqual(session.step_id, "identify")

            advance = await builder.get_function("workflow__advance")
            with invocation_scope(session, "advance-trace"):
                response = await advance.ainvoke({"skip": False}, to_type=str)
            self.assertEqual(session.step_id, "fill_water")
            self.assertEqual(response, workflow.step("fill_water").enter_message)
            reset = await builder.get_function("workflow__reset")
            with invocation_scope(session, "reset-trace"):
                await reset.ainvoke({"scope": "tea_guide"}, to_type=str)
            self.assertEqual(desktop.current(session), "root")

            with invocation_scope(session, "final-start"):
                await start.ainvoke({"scope": "tea_guide"}, to_type=str)
            session.step_id = "steep_timer"
            session.state["steeping_complete"] = True
            with invocation_scope(session, "final-advance"):
                await advance.ainvoke({"skip": False}, to_type=str)
            self.assertFalse(session.active)
            self.assertEqual(desktop.current(session), "root")

    async def test_background_controls_leave_the_root_in_foreground(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        desktop = DesktopRuntime(load_desktop(_APPLICATIONS))
        session = SessionStore(workflow).get("tester")

        async def notice(*_args) -> None:
            return None

        app = ChangeWatchApplication(
            desktop.spec.application("change_watch"),
            desktop,
            notice,
        )
        async with WorkflowBuilder() as builder:
            await add_background_controls(builder, app)
            start = await builder.get_function("change_watch__start")
            stop = await builder.get_function("change_watch__stop")
            status = await builder.get_function("change_watch__status")
            with invocation_scope(session, "start-background"):
                response = await start.ainvoke(
                    {"instruction": "people entering the doorway"},
                    to_type=str,
                )
            self.assertIn("started in the background", response)
            self.assertIn("people entering the doorway", response)
            self.assertEqual(
                app._states[session.participant_id].instruction,
                "people entering the doorway",
            )
            with invocation_scope(session, "status-background"):
                response = await status.ainvoke({}, to_type=str)
            self.assertIn("Monitoring: people entering the doorway", response)
            self.assertEqual(desktop.current(session), "root")
            self.assertTrue(desktop.is_background_active(session, "change_watch"))

            with invocation_scope(session, "stop-background"):
                response = await stop.ainvoke({}, to_type=str)
            self.assertIn("stopped", response)
            self.assertEqual(desktop.current(session), "root")


if __name__ == "__main__":
    unittest.main()
