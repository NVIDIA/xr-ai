# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tea_making_worker.agents.registry import AgentRegistry, _state_contract
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


class _Capture:
    request = ""

    async def ainvoke(self, request, *, to_type):
        self.request = request
        return "captured"


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


if __name__ == "__main__":
    unittest.main()
