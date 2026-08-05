# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from tea_making_worker.functions import CurrentViewConfig, CurrentViewRequest, add_workflow_functions
from tea_making_worker.functions.vision import current_view
from tea_making_worker.runtime.scope import invocation_scope
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow
from xr_ai_nat.functions.vision import LiveVisionResult

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


class NatExecutionTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_workflow_nat_functions_are_the_only_mutation_surface(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        session = store.get("tester")

        async def answer_step(*_args) -> str:
            return "step answer"

        async with WorkflowBuilder() as builder:
            await add_workflow_functions(builder, store=store, answer_step=answer_step)
            start = await builder.get_function("workflow__start")
            commit = await builder.get_function("workflow__commit")
            with invocation_scope(session, "route-trace"):
                response = await start.ainvoke({}, to_type=str)
            self.assertEqual(response, workflow.step("identify").enter_message)
            self.assertEqual(session.step_id, "identify")
            for index in range(2):
                store.observe(
                    session,
                    "A tea package label reads Oolong, 88 C, steep 4 minutes.",
                    f"identify-{index}",
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


if __name__ == "__main__":
    unittest.main()
