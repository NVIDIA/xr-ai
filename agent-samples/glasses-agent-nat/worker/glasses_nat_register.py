# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register glasses-agent NAT function groups."""

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionGroupBaseConfig

from glasses_nat_schemas import AnalyzeRecordingInput
from glasses_nat_schemas import CondenseObservationsInput
from glasses_nat_schemas import DeriveStepKeyInfoInput
from glasses_nat_schemas import DeriveStepRequirementsInput
from glasses_nat_schemas import GuidanceStepInput
from glasses_nat_tasks import analyze_recording_impl
from glasses_nat_tasks import check_guidance_step_complete_impl
from glasses_nat_tasks import condense_observations_impl
from glasses_nat_tasks import derive_step_key_info_impl
from glasses_nat_tasks import describe_current_view_impl
from glasses_nat_tasks import derive_step_requirements_impl


class GlassesAgentToolsConfig(FunctionGroupBaseConfig, name="glasses_agent_tools"):
    """LLM-facing tools for the glasses request-time NAT agent."""


class GlassesWorkerTasksConfig(FunctionGroupBaseConfig, name="glasses_worker_tasks"):
    """Worker-internal NAT tasks for bounded LLM/tool work.

    LLM endpoints are not configured here: the tasks resolve their models from
    the workflow ``llms:`` block via ``builder.get_llm(...)``, so model config,
    retries, and timeouts live in one place.

    The fields are typed ``LLMRef`` (not ``str``) so NAT's dependency resolver
    records this group's edge to those LLMs and builds them *before* this group
    — otherwise ``get_llm`` runs before the LLM exists and raises
    ``LLM `agent_llm` not found`` (the agent LLM is otherwise built lazily as
    the workflow's ``llm_name``, after function groups).
    """

    agent_llm_name: LLMRef = Field(
        default=LLMRef("agent_llm"),
        description="Name of the agentic LLM in the workflow `llms:` block.",
    )
    worker_llm_name: LLMRef = Field(
        default=LLMRef("worker_llm"),
        description="Name of the smaller worker LLM in the workflow `llms:` block.",
    )


@register_function_group(config_type=GlassesAgentToolsConfig)
async def glasses_agent_tools(config: GlassesAgentToolsConfig, builder: Builder):
    get_latest_frame = await builder.get_function("video_mcp__get_latest_frame")
    list_live_participants = await builder.get_function("video_mcp__list_live_participants")
    ask_image = await builder.get_function("vlm_mcp__ask_image")
    group = FunctionGroup(config=config)

    async def describe_current_view(participant_id: str, question: str = "") -> str:
        """Describe the current live camera view for one participant."""
        return await describe_current_view_impl(
            participant_id=participant_id,
            question=question,
            get_latest_frame=get_latest_frame,
            list_live_participants=list_live_participants,
            ask_image=ask_image,
        )

    group.add_function(
        "describe_current_view",
        describe_current_view,
        description=describe_current_view.__doc__,
    )
    yield group


@register_function_group(config_type=GlassesWorkerTasksConfig)
async def glasses_worker_tasks(config: GlassesWorkerTasksConfig, builder: Builder):
    get_latest_frame = await builder.get_function("video_mcp__get_latest_frame")
    ask_image = await builder.get_function("vlm_mcp__ask_image")
    ask_frames = await builder.get_function("vlm_mcp__ask_frames")
    # Bounded worker-task LLM calls now go through NAT's LLM layer instead of
    # hand-rolled httpx, so endpoint/retry/timeout config lives in `llms:`.
    agent_llm = await builder.get_llm(
        config.agent_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )
    worker_llm = await builder.get_llm(
        config.worker_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )
    group = FunctionGroup(config=config)

    async def analyze_recording(request: AnalyzeRecordingInput) -> dict:
        """Analyze recorded frame and narration timelines into demo steps."""
        result = await analyze_recording_impl(request, agent_llm=agent_llm)
        return result.model_dump()

    async def condense_observations(request: CondenseObservationsInput) -> dict:
        """Condense recent observations into a scene summary and event list."""
        result = await condense_observations_impl(request, worker_llm=worker_llm)
        return result.model_dump()

    async def check_guidance_step_complete(request: GuidanceStepInput) -> dict:
        """Return whether the current live frame satisfies a guidance step."""
        result = await check_guidance_step_complete_impl(
            participant_id=request.participant_id,
            instruction=request.instruction,
            expected_requirements=list(request.expected_requirements),
            teacher_image_path=request.teacher_image_path,
            teacher_caption=request.teacher_caption,
            min_live_timestamp_us=request.min_live_timestamp_us,
            key_objects=list(request.key_objects),
            key_action=request.key_action,
            key_position=request.key_position,
            key_target_state=request.key_target_state,
            key_ignore=list(request.key_ignore),
            get_latest_frame=get_latest_frame,
            ask_image=ask_image,
            ask_frames=ask_frames,
        )
        return result.model_dump()

    async def derive_step_requirements(request: DeriveStepRequirementsInput) -> dict:
        """Derive an atomic visual checklist for one analyzed step."""
        result = await derive_step_requirements_impl(request, agent_llm=agent_llm)
        return result.model_dump()

    async def derive_step_key_info(request: DeriveStepKeyInfoInput) -> dict:
        """Distill one step into structured key info (objects/action/position/...)."""
        result = await derive_step_key_info_impl(request, agent_llm=agent_llm)
        return result.model_dump()

    group.add_function(
        "analyze_recording",
        analyze_recording,
        input_schema=AnalyzeRecordingInput,
        description=analyze_recording.__doc__,
    )
    group.add_function(
        "condense_observations",
        condense_observations,
        input_schema=CondenseObservationsInput,
        description=condense_observations.__doc__,
    )
    group.add_function(
        "check_guidance_step_complete",
        check_guidance_step_complete,
        input_schema=GuidanceStepInput,
        description=check_guidance_step_complete.__doc__,
    )
    group.add_function(
        "derive_step_requirements",
        derive_step_requirements,
        input_schema=DeriveStepRequirementsInput,
        description=derive_step_requirements.__doc__,
    )
    group.add_function(
        "derive_step_key_info",
        derive_step_key_info,
        input_schema=DeriveStepKeyInfoInput,
        description=derive_step_key_info.__doc__,
    )
    yield group
