# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from tea_making_worker.desktop.runtime import DesktopRuntime
from tea_making_worker.desktop.spec import load_desktop
from tea_making_worker.engine.coordinator import Coordinator
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow
from xr_ai_voice import VoiceTurn

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"
_APPLICATIONS = Path(__file__).parents[2] / "yaml" / "applications.yaml"


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


class _Desktop:
    def __init__(self) -> None:
        self.runtime = DesktopRuntime(load_desktop(_APPLICATIONS))

    async def route(self, *_args) -> str:
        return "desktop answer"


class _Backgrounds:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.releases = 0

    async def on_transcription(self, _session, text, _trace_id) -> None:
        self.inputs.append(text)

    async def tick(self, _session) -> None:
        return None

    async def release(self, _session) -> None:
        self.releases += 1


async def _notice(*_args) -> None:
    pass


def _coordinator(store, *, agents=None, triggers=None):
    desktop = _Desktop()
    backgrounds = _Backgrounds()
    return (
        Coordinator(
            store=store,
            agents=agents or _Agents(),
            desktop=desktop,
            backgrounds=backgrounds,
            triggers=triggers or _Triggers(),
            notice=_notice,
        ),
        desktop,
        backgrounds,
    )


class CoordinatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_raw_transcription_reaches_background_without_a_routed_query(self) -> None:
        store = SessionStore(load_workflow(_WORKFLOW))
        coordinator, _, backgrounds = _coordinator(store)

        await coordinator.handle_transcription(
            VoiceTurn(
                participant_id="tester",
                role="user",
                timestamp_us=123,
                text="ordinary speech without a wake word",
            )
        )

        self.assertEqual(backgrounds.inputs, ["ordinary speech without a wake word"])

    async def test_routed_query_is_not_recorded_twice(self) -> None:
        store = SessionStore(load_workflow(_WORKFLOW))
        coordinator, _, backgrounds = _coordinator(store)

        self.assertEqual(await coordinator.handle_query("tester", "agent status"), "desktop answer")

        self.assertEqual(backgrounds.inputs, [])

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
        coordinator, _, _ = _coordinator(store, agents=agents, triggers=triggers)

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
        coordinator, desktop, _ = _coordinator(store)
        desktop.runtime.capture(session, "tea")
        desktop.runtime.start_background(session, "change_watch")

        await coordinator.participant_joined("tester")

        self.assertFalse(session.active)
        self.assertIsNone(session.step_id)
        self.assertNotIn("tea_name", session.state)
        self.assertEqual(desktop.runtime.current(session), "root")
        self.assertEqual(session.desktop.background, set())

    async def test_roster_replay_does_not_reset_an_active_connection(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        coordinator, desktop, _ = _coordinator(store)
        await coordinator.participant_joined("tester")
        session = store.get("tester")
        store.start(session)
        desktop.runtime.capture(session, "tea")

        await coordinator.participant_joined("tester")

        self.assertTrue(session.active)
        self.assertEqual(session.step_id, "identify")
        self.assertEqual(desktop.runtime.current(session), "tea")

    async def test_reconnect_after_leave_gets_a_fresh_session(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        store = SessionStore(workflow)
        coordinator, desktop, _ = _coordinator(store)
        await coordinator.participant_joined("tester")
        session = store.get("tester")
        store.start(session)
        desktop.runtime.capture(session, "tea")
        session.state["tea_name"] = "stale tea"
        await coordinator.participant_left("tester")

        await coordinator.participant_joined("tester")
        fresh = store.get("tester")

        self.assertFalse(fresh.active)
        self.assertIsNone(fresh.step_id)
        self.assertNotIn("tea_name", fresh.state)
        self.assertEqual(desktop.runtime.current(fresh), "root")


if __name__ == "__main__":
    unittest.main()
