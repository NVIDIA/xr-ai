# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register glasses-agent NAT function groups."""

from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

from xr_ai_models import load_models_config, make_llm

from glasses_nat_schemas import AnalyzeRecordingInput
from glasses_nat_schemas import CondenseObservationsInput
from glasses_nat_schemas import GuidanceStepInput
from glasses_nat_tasks import analyze_recording_impl
from glasses_nat_tasks import check_guidance_step_complete_impl
from glasses_nat_tasks import condense_observations_impl
from glasses_nat_tasks import describe_current_view_impl


class GlassesAgentToolsConfig(FunctionGroupBaseConfig, name="glasses_agent_tools"):
    """LLM-facing tools for the glasses request-time NAT agent."""


class GlassesWorkerTasksConfig(FunctionGroupBaseConfig, name="glasses_worker_tasks"):
    """Worker-internal NAT tasks for bounded LLM/tool work.

    LLM endpoints come from a `models.yaml` (per AGENTS.md: all AI-service
    HTTP goes through ``agent-sdk/xr-ai-models``). The yaml is resolved
    relative to the NAT workflow YAML file's directory unless an absolute
    path is given.
    """

    models_yaml: str = Field(
        default="models.yaml",
        description=(
            "Path to the sample's models.yaml. Resolved relative to the "
            "NAT workflow YAML directory when not absolute."
        ),
    )
    agent_llm_name: str = Field(
        default="agent_llm",
        description="Logical name inside models.yaml for the reasoning-capable LLM.",
    )
    worker_llm_name: str = Field(
        default="worker_llm",
        description="Logical name inside models.yaml for the smaller / faster LLM.",
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


def _resolve_models_yaml(config: GlassesWorkerTasksConfig, builder: Builder) -> Path:
    p = Path(config.models_yaml)
    if p.is_absolute():
        return p
    # NAT workflow files set ``builder.workflow_config_file`` (or similar) at
    # load time. Fall back to CWD if unavailable so single-process usage and
    # tests still work.
    workflow_file = getattr(builder, "workflow_config_file", None) or getattr(builder, "config_file", None)
    base = Path(workflow_file).parent if workflow_file else Path.cwd()
    return base / p


@register_function_group(config_type=GlassesWorkerTasksConfig)
async def glasses_worker_tasks(config: GlassesWorkerTasksConfig, builder: Builder):
    get_latest_frame = await builder.get_function("video_mcp__get_latest_frame")
    ask_image = await builder.get_function("vlm_mcp__ask_image")

    models_cfg = load_models_config(_resolve_models_yaml(config, builder))
    agent_llm = make_llm(models_cfg, config.agent_llm_name)
    worker_llm = make_llm(models_cfg, config.worker_llm_name)

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
            get_latest_frame=get_latest_frame,
            ask_image=ask_image,
        )
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
    try:
        yield group
    finally:
        # LLM clients hold long-lived HTTP connections — release on workflow shutdown.
        await agent_llm.close()
        await worker_llm.close()
