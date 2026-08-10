# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from tea_making_worker.applications.context import (
    ApplicationContextFunctionsConfig,
    ContextQueryRequest,
    ContextQueryResult,
)
from tea_making_worker.applications.context.functions import ContextClearRequest, ContextClearResult
from tea_making_worker.applications.events import BACKGROUND_FACT, BackgroundFact
from tea_making_worker.runtime.scope import current_invocation, invocation_scope
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


class ApplicationContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_context_is_participant_scoped_bounded_and_consumer_selected(self) -> None:
        sessions = SessionStore(load_workflow(_WORKFLOW))
        first = sessions.get("first")
        second = sessions.get("second")

        async with WorkflowBuilder() as builder:
            group = await builder.add_function_group(
                "context_store",
                ApplicationContextFunctionsConfig(capacity_per_participant=3),
            )
            functions = await group.get_all_functions()
            record = functions["context_store__record"]
            query = functions["context_store__query"]
            clear = functions["context_store__clear"]

            for index, topic in enumerate(
                ("transcript.summary", "video_log.delta", "change_watch.change", "video_log.delta")
            ):
                await record.ainvoke(
                    BACKGROUND_FACT.envelope(
                        participant_id=first.participant_id,
                        producer="test",
                        payload=BackgroundFact(topic=topic, summary=f"fact {index}"),
                    )
                )

            with invocation_scope(second, "query-second"):
                empty = await query.ainvoke(ContextQueryRequest(), to_type=ContextQueryResult)
            self.assertEqual(empty.items, ())

            with invocation_scope(first, "query-first"):
                selected = await query.ainvoke(
                    ContextQueryRequest(topics=("video_log.delta",), max_items=1),
                    to_type=ContextQueryResult,
                )
                operation = current_invocation().route_operation

            self.assertEqual([item.summary for item in selected.items], ["fact 3"])
            self.assertEqual(operation, "application_context.query")

            with invocation_scope(first, "clear-first"):
                cleared = await clear.ainvoke(ContextClearRequest(), to_type=ContextClearResult)
                after_clear = await query.ainvoke(ContextQueryRequest(), to_type=ContextQueryResult)

            self.assertEqual(cleared.removed, 3)
            self.assertEqual(after_clear.items, ())


if __name__ == "__main__":
    unittest.main()
