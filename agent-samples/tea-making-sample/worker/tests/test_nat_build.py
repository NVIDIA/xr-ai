# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from tea_making_worker.agents import AgentRegistry
from tea_making_worker.agents.factory import add_guidance_llm
from tea_making_worker.applications.compose import build_applications
from tea_making_worker.applications.context import ApplicationContextFunctionsConfig, add_context_query
from tea_making_worker.applications.events import BACKGROUND_FACT
from tea_making_worker.applications.manager.runtime import ApplicationOwnership
from tea_making_worker.applications.manager.spec import load_application_catalog
from tea_making_worker.applications.manager.turn import ApplicationTurn
from tea_making_worker.applications.output import UserOutputDelivery
from tea_making_worker.functions import (
    CurrentViewConfig,
    RAGLookupConfig,
    add_clock_functions,
    add_temperature_functions,
    add_workflow_functions,
)
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow
from xr_ai_models import Capabilities, ChatResponse
from xr_ai_nat.events import EventDispatcher, add_event_handler
from xr_ai_nat.functions.rag import RAGFunctionsConfig
from xr_ai_nat.functions.vision import VisionToolsConfig

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"
_APPLICATIONS = Path(__file__).parents[2] / "yaml" / "applications.yaml"


class _Endpoint:
    def on_frame(self, callback):
        self.frame_callback = callback

    def on_participant(self, callback):
        self.participant_callback = callback


class _LLM:
    capabilities = Capabilities(tool_calls=True)

    def __init__(self) -> None:
        self.responses: list[ChatResponse] = []

    async def chat(self, *_args, **_kwargs):
        return self.responses.pop(0)

    async def close(self):
        pass


class _VLM:
    capabilities = Capabilities(vision=True)


class _Transport:
    async def send_return_data(self, _message) -> None:
        return None


class _VoiceSession:
    transport = _Transport()

    async def enqueue_response(self, *_args, **_kwargs) -> None:
        return None


class NatBuildTest(unittest.IsolatedAsyncioTestCase):
    async def test_every_yaml_tool_resolves_during_nat_agent_build(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        application_spec = load_application_catalog(_APPLICATIONS)
        application_runtime = ApplicationOwnership(application_spec)
        store = SessionStore(workflow)
        agents = AgentRegistry(workflow)
        llm = _LLM()
        async with WorkflowBuilder() as builder:
            vision_group = await builder.add_function_group(
                "vision",
                VisionToolsConfig(endpoint=_Endpoint(), vlm=_VLM()),
            )
            functions = await vision_group.get_all_functions()
            current_view = await builder.add_function(
                "current_view",
                CurrentViewConfig(source=functions["vision__look_at_current_frame"]),
            )
            rag_group = await builder.add_function_group(
                "rag",
                RAGFunctionsConfig(endpoint="tcp://127.0.0.1:1"),
            )
            rag_functions = await rag_group.get_all_functions()
            await builder.add_function(
                "rag_lookup",
                RAGLookupConfig(source=rag_functions["rag__retrieve"]),
            )
            await add_clock_functions(builder)
            await add_temperature_functions(builder)
            await add_workflow_functions(
                builder,
                store=store,
                application_ownership=application_runtime,
            )
            context_group = await builder.add_function_group(
                "context_store",
                ApplicationContextFunctionsConfig(),
            )
            context_functions = await context_group.get_all_functions()
            events = EventDispatcher()
            events.subscribe(
                BACKGROUND_FACT,
                subscriber_id="context.recorder",
                function=context_functions["context_store__record"],
            )

            async def reset_context(_event) -> None:
                await context_functions["context_store__clear"].ainvoke({})

            context_reset = await add_event_handler(
                builder,
                name="application__reset_context",
                handler=reset_context,
                description="Clear participant context when applications reset.",
            )
            self.assertIs(
                context_reset,
                await builder.get_function("application__reset_context"),
            )
            await add_context_query(builder, context_functions["context_store__query"])
            output = UserOutputDelivery(events, _VoiceSession())  # type: ignore[arg-type]
            await output.build(builder)
            llm_ref = await add_guidance_llm(builder, llm)
            await agents.build(builder, llm_ref)
            applications = await build_applications(
                builder,
                llm_ref=llm_ref,
                spec=application_spec,
                ownership=application_runtime,
                tea=agents,
                current_view=current_view,
                events=events,
                output=output,
                store=store,
            )
            self.assertIs(
                applications.manager.function,
                await builder.get_function("application_manager__turn"),
            )
            self.assertIs(
                applications.manager._foreground["tea"],
                await builder.get_function("application__tea_turn"),
            )
            self.assertIs(applications.manager.function.input_schema, ApplicationTurn)
            self.assertEqual(len(applications.periodic_sources), 3)
            self.assertIs(applications.change_watch.periodic, applications.periodic_sources[0])
            self.assertIs(applications.transcript.periodic, applications.periodic_sources[1])
            self.assertIs(applications.video_log.periodic, applications.periodic_sources[2])
            self.assertEqual(
                {function.name for function in applications.manager.functions},
                {
                    "current_view",
                    "rag_lookup",
                    "application_context__query",
                    "workflow__start",
                    "application_manager__status",
                    "change_watch__start",
                    "change_watch__stop",
                    "change_watch__status",
                    "transcript__start",
                    "transcript__stop",
                    "transcript__status",
                    "video_log__start",
                    "video_log__stop",
                    "video_log__status",
                },
            )


if __name__ == "__main__":
    unittest.main()
