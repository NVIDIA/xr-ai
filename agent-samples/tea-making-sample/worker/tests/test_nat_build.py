# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from tea_making_worker.agents import AgentRegistry
from tea_making_worker.agents.factory import add_guidance_llm
from tea_making_worker.applications.compose import build_applications
from tea_making_worker.desktop.runtime import DesktopRuntime
from tea_making_worker.desktop.spec import load_desktop
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
from xr_ai_nat.functions.rag import RAGFunctionsConfig
from xr_ai_nat.functions.vision import VisionToolsConfig

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"
_APPLICATIONS = Path(__file__).parents[2] / "yaml" / "applications.yaml"


async def _notice(*_args) -> None:
    return None


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


class NatBuildTest(unittest.IsolatedAsyncioTestCase):
    async def test_every_yaml_tool_resolves_during_nat_agent_build(self) -> None:
        workflow = load_workflow(_WORKFLOW)
        desktop_spec = load_desktop(_APPLICATIONS)
        desktop_runtime = DesktopRuntime(desktop_spec)
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
            await add_workflow_functions(builder, store=store, desktop=desktop_runtime)
            llm_ref = await add_guidance_llm(builder, llm)
            await agents.build(builder, llm_ref)
            applications = await build_applications(
                builder,
                llm_ref=llm_ref,
                spec=desktop_spec,
                runtime=desktop_runtime,
                tea=agents,
                current_view=current_view,
                notice=_notice,
                text_output=_notice,
            )
            self.assertEqual(
                {function.name for function in applications.desktop.functions},
                {
                    "current_view",
                    "rag_lookup",
                    "workflow__start",
                    "desktop__status",
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
