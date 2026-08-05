# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable execution mechanisms for YAML-defined workflow steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from loguru import logger

from .agent import StepAgentResult, WorkflowAgent
from .vision import StepVision
from .workflow import WorkflowDefinition, WorkflowStep

StepEventKind = Literal["periodic", "voice"]


@dataclass(frozen=True, slots=True)
class StepEvent:
    """One trigger delivered to the active step mechanism."""

    kind: StepEventKind
    transcript: str = ""
    recent_turns: tuple[tuple[str, str], ...] = ()
    step_state: str = "started"
    ready: bool = False
    workflow_active: bool = True

    @classmethod
    def periodic(cls, *, step_state: str = "started") -> StepEvent:
        return cls(kind="periodic", step_state=step_state)

    @classmethod
    def voice(
        cls,
        transcript: str,
        *,
        recent_turns: list[tuple[str, str]],
        step_state: str,
        ready: bool,
        workflow_active: bool,
    ) -> StepEvent:
        return cls(
            kind="voice",
            transcript=transcript,
            recent_turns=tuple(recent_turns),
            step_state=step_state,
            ready=ready,
            workflow_active=workflow_active,
        )


@dataclass(frozen=True, slots=True)
class MechanismObservation:
    """Visual evidence collected while handling a step event."""

    text: str
    frame_pts_us: int
    kind: str = "step_monitor"
    question: str = ""


@dataclass(frozen=True, slots=True)
class StepIteration:
    """Output from one mechanism iteration."""

    result: StepAgentResult
    observations: tuple[MechanismObservation, ...] = ()


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
        event: StepEvent,
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
        event: StepEvent,
    ) -> StepIteration | None:
        if event.kind == "voice":
            return await self._run_voice(
                participant_id=participant_id,
                step=step,
                context=context,
                observation_log=observation_log,
                event=event,
            )

        caption = ""
        frame_pts_us: int | None = None
        vision_stopped = self._workflow.condition_met(step.vlm_stop_when, context)
        if step.vlm_prompt.strip() and not vision_stopped:
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
            "step mechanism agent begin pid={} step={} mechanism={} has_caption={} vision_stopped={}",
            participant_id,
            step.id,
            self.name,
            bool(caption),
            vision_stopped,
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
            observations=(MechanismObservation(caption, frame_pts_us),) if caption and frame_pts_us is not None else (),
        )

    async def _run_voice(
        self,
        *,
        participant_id: str,
        step: WorkflowStep,
        context: dict[str, Any],
        observation_log: list[dict[str, Any]],
        event: StepEvent,
    ) -> StepIteration:
        observations: list[MechanismObservation] = []

        async def inspect_current_view(question: str) -> dict[str, Any]:
            if not hasattr(self._vision, "inspect"):
                return {"error": "A live camera view is not available."}
            observation = await self._vision.inspect(
                participant_id,
                question,
                step=step,
                context=context,
                task=self._workflow.task,
            )
            observations.append(
                MechanismObservation(
                    text=observation.text,
                    frame_pts_us=observation.frame_pts_us,
                    kind="visual_question",
                    question=question,
                )
            )
            logger.info(
                "guide visual question pid={} step={} question={!r} answer={!r}",
                participant_id,
                step.id,
                question,
                observation.text[:500],
            )
            return {
                "question": question,
                "visual_evidence": observation.text,
                "frame_pts_us": observation.frame_pts_us,
            }

        answer = await self._agent.answer_step(
            transcript=event.transcript,
            context=context,
            current_step=step,
            step_state=event.step_state,
            ready=event.ready,
            workflow_active=event.workflow_active,
            observation_log=observation_log,
            recent_turns=list(event.recent_turns),
            visual_query=inspect_current_view,
        )

        return StepIteration(
            result=StepAgentResult(
                context_patch={},
                step_state=event.step_state,
                ready_to_advance=event.ready,
                assistant_message=answer,
            ),
            observations=tuple(observations),
        )


class StepMechanisms:
    """Resolve YAML mechanism names without coupling the guide to their logic."""

    def __init__(self, mechanisms: list[StepMechanism]) -> None:
        self._by_name = {mechanism.name: mechanism for mechanism in mechanisms}
        if len(self._by_name) != len(mechanisms):
            raise ValueError("step mechanism names must be unique")

    def validate(self, workflow: WorkflowDefinition) -> None:
        unknown = {
            step.mechanism for step in workflow.steps if not step.is_idle and step.mechanism not in self._by_name
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
        event: StepEvent,
    ) -> StepIteration | None:
        logger.debug(
            "step event pid={} step={} mechanism={} kind={} state={} ready={}",
            participant_id,
            step.id,
            step.mechanism,
            event.kind,
            event.step_state,
            event.ready,
        )
        return await self._by_name[step.mechanism].run(
            participant_id=participant_id,
            step=step,
            context=context,
            observation_log=observation_log,
            last_frame_pts_us=last_frame_pts_us,
            event=event,
        )


__all__ = [
    "CaptionAgentStepMechanism",
    "MechanismObservation",
    "StepEvent",
    "StepIteration",
    "StepMechanism",
    "StepMechanisms",
]
