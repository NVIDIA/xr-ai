# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only focused NAT agent for task guidance and grounded questions."""

from pathlib import Path

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionGroupRef, FunctionInfo, LLMRef, register_function
from nat.tool.chat_completion import ChatCompletionConfig
from xr_ai_nat.functions.rag import RetrieveResult

from .models import GuideAgentRequest, TaskGuideReply

_PROMPT = Path(__file__).with_name("prompts") / "guide_agent.txt"


class TaskGuideAgentConfig(FunctionBaseConfig, name="visual_task_guide_agent"):
    llm_name: LLMRef = LLMRef("guide_llm")
    task_state: FunctionGroupRef = FunctionGroupRef("task_state")
    task_knowledge: FunctionGroupRef = FunctionGroupRef("task_knowledge")


@register_function(config_type=TaskGuideAgentConfig)
async def task_guide_agent(config: TaskGuideAgentConfig, builder: Builder):
    state_group = await builder.get_function_group(config.task_state)
    state_functions = await state_group.get_all_functions()
    get_status = state_functions[f"{state_group.instance_name}__get_task_status"]
    knowledge_group = await builder.get_function_group(config.task_knowledge)
    knowledge_functions = await knowledge_group.get_all_functions()
    retrieve = knowledge_functions[f"{knowledge_group.instance_name}__retrieve"]
    reasoning = await builder.add_function(
        "task_guide_reasoning",
        ChatCompletionConfig(
            llm_name=config.llm_name,
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
        ),
    )

    async def guide(request: GuideAgentRequest) -> TaskGuideReply:
        status = await get_status.ainvoke({"participant_id": request.participant_id})
        knowledge = RetrieveResult.model_validate(
            await retrieve.ainvoke({"query": request.user_text, "top_k": 2})
        )
        context = "\n".join(
            f"[{result.source}] {result.text}" for result in knowledge.results
        ) or "No relevant task documentation was retrieved."
        message = (
            f"Trusted task state: {status.progress.state}\n"
            f"Current step: {status.current_step}\n"
            f"User request: {request.user_text}\n"
            f"Latest live observation: {request.latest_observation}\n"
            f"Retrieved task knowledge:\n{context}"
        )
        output = await reasoning.ainvoke(message, to_type=str)
        return TaskGuideReply(response=str(output or "I could not produce task guidance."))

    yield FunctionInfo.from_fn(
        guide,
        description="Answer task questions from read-only state, documentation, and supplied visual evidence.",
    )


__all__ = ["TaskGuideAgentConfig"]
