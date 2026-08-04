# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable execution mechanisms for YAML-defined workflow steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from .agent import StepAgentResult, WorkflowAgent
from .vision import StepVision
from .workflow import WorkflowDefinition, WorkflowStep


@dataclass(frozen=True, slots=True)
class StepIteration:
    """Output from one mechanism iteration."""

    result: StepAgentResult
    caption: str = ""
    frame_pts_us: int | None = None


class StepMechanism(Protocol):
    """A pluggable strategy that produces one step-agent result."""

    name: str

    async def run(
        self,
        *,
        participant_id: str,
        step: WorkflowStep,
        context: dict[str, Any],
        observation_log: list[dict[str, Any]],
        last_frame_pts_us: int,
    ) -> StepIteration | None: ...


class CaptionAgentStepMechanism:
    """Optionally caption a frame, then pass all interpretation to the agent."""

    name = "caption_agent"

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        vision: StepVision,
        agent: WorkflowAgent,
    ) -> None:
        self._workflow = workflow
        self._vision = vision
        self._agent = agent

    async def run(
        self,
        *,
        participant_id: str,
        step: WorkflowStep,
        context: dict[str, Any],
        observation_log: list[dict[str, Any]],
        last_frame_pts_us: int,
    ) -> StepIteration | None:
        caption = ""
        frame_pts_us: int | None = None
        if step.vlm_prompt.strip():
            observation = await self._vision.observe(
                participant_id,
                step,
                context,
                task=self._workflow.task,
            )
            if observation.frame_pts_us <= last_frame_pts_us:
                return None
            caption = observation.text
            frame_pts_us = observation.frame_pts_us

        logger.debug(
            "step mechanism agent begin pid={} step={} mechanism={} has_caption={}",
            participant_id,
            step.id,
            self.name,
            bool(caption),
        )
        result = await self._agent.run_step(
            step=step,
            participant_id=participant_id,
            context=context,
            observation_log=observation_log,
            vlm_observation=caption,
        )
        return StepIteration(
            result=result,
            caption=caption,
            frame_pts_us=frame_pts_us,
        )


class StepMechanisms:
    """Resolve YAML mechanism names without coupling the guide to their logic."""

    def __init__(self, mechanisms: list[StepMechanism]) -> None:
        self._by_name = {mechanism.name: mechanism for mechanism in mechanisms}
        if len(self._by_name) != len(mechanisms):
            raise ValueError("step mechanism names must be unique")

    def validate(self, workflow: WorkflowDefinition) -> None:
        unknown = {
            step.mechanism
            for step in workflow.steps
            if not step.is_idle and step.mechanism not in self._by_name
        }
        if unknown:
            raise ValueError(f"unknown step mechanisms: {', '.join(sorted(unknown))}")

    async def run(
        self,
        *,
        participant_id: str,
        step: WorkflowStep,
        context: dict[str, Any],
        observation_log: list[dict[str, Any]],
        last_frame_pts_us: int,
    ) -> StepIteration | None:
        return await self._by_name[step.mechanism].run(
            participant_id=participant_id,
            step=step,
            context=context,
            observation_log=observation_log,
            last_frame_pts_us=last_frame_pts_us,
        )


__all__ = [
    "CaptionAgentStepMechanism",
    "StepIteration",
    "StepMechanism",
    "StepMechanisms",
]
