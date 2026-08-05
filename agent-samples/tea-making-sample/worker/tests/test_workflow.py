# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


class WorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow(_WORKFLOW)
        self.store = SessionStore(self.workflow)
        self.session = self.store.get("tester")

    def _observe_identification(self) -> None:
        self.store.observe(
            self.session,
            "A tea package label reads Oolong, 88 C, steep 4 minutes.",
            "identify",
        )

    def test_all_steps_use_the_same_declarative_contract(self) -> None:
        self.assertEqual(self.workflow.start_step, "identify")
        self.assertEqual(len(self.workflow.steps), 5)
        for step in self.workflow.steps.values():
            self.assertTrue(step.trigger.function)
            self.assertTrue(step.agent.prompt)
            self.assertTrue(step.voice.prompt)
            self.assertTrue(step.complete_when)
            self.assertTrue(step.state_on_skip)
        identify = self.workflow.step("identify")
        self.assertEqual(identify.agent.tools, ("rag_lookup",))
        self.assertIn("rag_lookup", identify.voice.tools)

    def test_commit_marks_ready_but_voice_advance_transitions(self) -> None:
        self.store.start(self.session)
        self._observe_identification()
        result = self.store.commit(
            self.session,
            {
                "tea_name": "oolong",
                "target_temperature_c": 88,
                "steep_duration_s": 240,
                "guidance_source": "package",
                "tea_ready": True,
            },
            "",
        )
        self.assertTrue(result.accepted)
        self.assertTrue(result.complete)
        self.assertEqual(self.session.step_id, "identify")
        self.assertEqual(self.session.state["tea_name"], "oolong")
        self.assertTrue(self.store.drain_notices(self.session))
        revision = self.session.revision
        repeated = self.store.commit(
            self.session,
            {
                "tea_name": "black tea",
                "target_temperature_c": 88,
                "steep_duration_s": 240,
                "guidance_source": "package",
                "tea_ready": True,
            },
            "repeated notice",
        )
        self.assertTrue(repeated.complete)
        self.assertEqual(self.session.revision, revision)
        self.assertEqual(self.session.state["tea_name"], "oolong")
        self.assertFalse(self.store.drain_notices(self.session))
        response = self.store.advance(self.session, skip=False)
        self.assertEqual(self.session.step_id, "fill_water")
        self.assertEqual(response, self.workflow.step("fill_water").enter_message)

    def test_invalid_commit_is_atomic(self) -> None:
        self.store.start(self.session)
        before = dict(self.session.state)
        result = self.store.commit(
            self.session,
            {"tea_name": "black tea", "water_filled": True},
            "",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(self.session.state, before)

    def test_context_is_only_current_step_reads_and_writes(self) -> None:
        self.store.start(self.session)
        step = self.workflow.step("heat_water")
        self.session.state.update({"target_temperature_c": 80, "tea_name": "green"})
        self.assertEqual(
            self.workflow.project(step, self.session.state),
            {"target_temperature_c": 80, "heating_started": False, "water_ready": False},
        )

    def test_agent_message_announces_one_intermediate_change(self) -> None:
        self.store.start(self.session)
        self.store.advance(self.session, skip=True)
        self.store.advance(self.session, skip=True)

        result = self.store.commit(
            self.session,
            {"heating_started": True},
            "The water is heating.",
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.complete)
        self.assertEqual(self.store.drain_notices(self.session), ("The water is heating.",))

        revision = self.session.revision
        repeated = self.store.commit(self.session, {"heating_started": True}, "")
        self.assertTrue(repeated.accepted)
        self.assertEqual(self.session.revision, revision)
        self.assertFalse(self.store.drain_notices(self.session))

    def test_management_messages_render_natural_units(self) -> None:
        self.store.start(self.session)
        self.store.advance(self.session, skip=True)
        heat = self.store.advance(self.session, skip=True)
        self.assertIn("93 degrees Celsius", heat)
        self.store.advance(self.session, skip=True)
        timer = self.store.advance(self.session, skip=True)
        self.assertIn("3 minutes", timer)
        self.assertNotIn("180 seconds", timer)

    def test_completion_evidence_rejects_agent_inversion(self) -> None:
        self.store.start(self.session)
        self._observe_identification()
        self.store.commit(
            self.session,
            {
                "tea_name": "oolong",
                "target_temperature_c": 88,
                "steep_duration_s": 240,
                "guidance_source": "package",
                "tea_ready": True,
            },
            "",
        )
        self.store.advance(self.session, skip=False)

        self.store.observe(self.session, "No vessel interior or water is visible.", "trace-no")
        rejected = self.store.commit(self.session, {"water_filled": True}, "")
        self.assertFalse(rejected.accepted)
        for index in range(2):
            self.store.observe(
                self.session,
                "The kettle interior is visible with a clear water surface and level inside.",
                f"trace-{index}",
            )
            rejected = self.store.commit(self.session, {"water_filled": True}, "")
            self.assertFalse(rejected.accepted)

        self.store.observe(
            self.session,
            "The kettle interior is visible with a clear water surface and level inside.",
            "trace-3",
        )
        accepted = self.store.commit(self.session, {"water_filled": True}, "")
        self.assertTrue(accepted.accepted)
        self.assertTrue(accepted.complete)

    def test_identification_accepts_ocr_and_rejects_dark_frames(self) -> None:
        self.store.start(self.session)
        self.store.observe(
            self.session,
            "NUMI ORGANIC BLACK TEA BREAKFAST BLEND",
            "readable",
        )
        self.assertEqual(self.session.evidence_hits, 1)
        self.store.observe(
            self.session,
            "The image is too dark to discern tea package text.",
            "dark",
        )
        self.assertEqual(self.session.evidence_hits, 0)
        self.store.observe(
            self.session,
            "There are no visible texts on any objects within this frame.",
            "empty",
        )
        self.assertEqual(self.session.evidence_hits, 0)
        self.store.observe(self.session, "none", "none")
        self.assertEqual(self.session.evidence_hits, 0)


if __name__ == "__main__":
    unittest.main()
