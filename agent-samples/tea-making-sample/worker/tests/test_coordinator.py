# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from tea_making_worker.engine.coordinator import Coordinator
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


class _Triggers:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, *_args):
        self.calls += 1
        return "fresh observation"


class _Agents:
    def __init__(self) -> None:
        self.calls = 0

    async def observe(self, *_args):
        self.calls += 1


async def _notice(*_args) -> None:
    pass


class CoordinatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_completed_step_keeps_the_homogeneous_observation_loop(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        session = store.get("tester")
        store.start(session)
        store.observe(
            session,
            "A tea package label reads Oolong, 88 C, steep 4 minutes.",
            "identify",
        )
        store.commit(
            session,
            {
                "tea_name": "oolong",
                "target_temperature_c": 88,
                "steep_duration_s": 240,
                "guidance_source": "package",
                "tea_ready": True,
            },
            "",
        )
        store.drain_notices(session)
        triggers = _Triggers()
        agents = _Agents()
        coordinator = Coordinator(
            store=store,
            agents=agents,
            triggers=triggers,
            notice=_notice,
        )

        await coordinator._tick(session)

        self.assertEqual(triggers.calls, 1)
        self.assertEqual(agents.calls, 1)
        self.assertEqual(session.step_id, "identify")

    async def test_participant_join_always_resets_existing_session(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        session = store.get("tester")
        store.start(session)
        session.state["tea_name"] = "stale tea"
        coordinator = Coordinator(
            store=store,
            agents=_Agents(),
            triggers=_Triggers(),
            notice=_notice,
        )

        await coordinator.participant_joined("tester")

        self.assertFalse(session.active)
        self.assertIsNone(session.step_id)
        self.assertNotIn("tea_name", session.state)

    async def test_roster_replay_does_not_reset_an_active_connection(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        coordinator = Coordinator(
            store=store,
            agents=_Agents(),
            triggers=_Triggers(),
            notice=_notice,
        )
        await coordinator.participant_joined("tester")
        session = store.get("tester")
        store.start(session)

        await coordinator.participant_joined("tester")

        self.assertTrue(session.active)
        self.assertEqual(session.step_id, "identify")

    async def test_reconnect_after_leave_gets_a_fresh_session(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        coordinator = Coordinator(
            store=store,
            agents=_Agents(),
            triggers=_Triggers(),
            notice=_notice,
        )
        await coordinator.participant_joined("tester")
        session = store.get("tester")
        store.start(session)
        session.state["tea_name"] = "stale tea"
        await coordinator.participant_left("tester")

        await coordinator.participant_joined("tester")
        fresh = store.get("tester")

        self.assertFalse(fresh.active)
        self.assertIsNone(fresh.step_id)
        self.assertNotIn("tea_name", fresh.state)


if __name__ == "__main__":
    unittest.main()
