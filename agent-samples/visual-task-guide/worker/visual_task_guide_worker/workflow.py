# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NAT workflow for explicit task controls and grounded questions."""

import re
import time
from difflib import SequenceMatcher

from loguru import logger
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionGroupRef, FunctionInfo, FunctionRef, register_function
from xr_ai_nat.functions.vision import VisionRequest, VisionResult

from .finger_count import FingerCount, format_finger_count, parse_finger_count
from .models import GuideAgentRequest, TaskGuideReply, TaskGuideRequest, TaskStatusResult

_START = frozenset({"start", "start task", "start the task", "begin task"})
_RESET = frozenset({"reset", "reset task", "reset the task", "start over"})
_NEXT = frozenset({"next", "next step", "go to next step"})
_STATUS = frozenset({"status", "task status", "current step", "what is the current step"})
_NEXT_INFO = frozenset(
    {"what's the next step", "what is the next step", "tell me the next step", "which step is next"}
)
_VALIDATE = frozenset(
    {
        "did i do the step correctly",
        "did i do it correctly",
        "did i do it right",
        "is this correct",
        "am i showing the right number",
    }
)
_CONTROLS = {
    "start": _START,
    "reset": _RESET,
    "next": _NEXT,
    "status": _STATUS,
    "next_info": _NEXT_INFO,
    "validate": _VALIDATE,
}
_WAKE_PREFIXES = ("hey agent ", "agent ")
_COUNT_QUERY = (
    "Independently count every clearly visible extended finger in the current frame. "
    "Do not assume a target count and do not use task instructions as visual evidence."
)
_COUNT_QUESTION_MARKERS = ("how many finger", "count the finger", "finger count")


def _command(text: str) -> str:
    command = " ".join(re.sub(r"[^a-z0-9']+", " ", text.casefold()).split())
    if command in {"agent", "hey agent"}:
        return ""
    for prefix in _WAKE_PREFIXES:
        if command.startswith(prefix):
            return command.removeprefix(prefix).strip()
    return command


def _control_command(command: str) -> str | None:
    if any(phrase in command for phrase in _VALIDATE):
        return "validate"
    for control, phrases in _CONTROLS.items():
        if command in phrases:
            return control
    if not command or len(command.split()) > 4:
        return None
    scores = sorted(
        (
            (SequenceMatcher(None, command, phrase).ratio(), control)
            for control, phrases in _CONTROLS.items()
            for phrase in phrases
        ),
        reverse=True,
    )
    best_score, best_control = scores[0]
    second_score = next(score for score, control in scores[1:] if control != best_control)
    return best_control if best_score >= 0.84 and best_score - second_score >= 0.04 else None


def format_task_status(status: TaskStatusResult) -> str:
    step = status.current_step
    if status.progress.state == "completed":
        return "Task complete. Say “reset task” to start over."
    if status.progress.state == "not_started":
        assert step is not None
        return f"Ready: {step.title}. Say “start task”."
    assert step is not None
    return f"{step.title}: {step.instructions}"


class TaskGuideWorkflowConfig(FunctionBaseConfig, name="visual_task_guide_workflow"):
    task_state: FunctionGroupRef = FunctionGroupRef("task_state")
    task_control: FunctionGroupRef = FunctionGroupRef("task_control")
    vision: FunctionRef = FunctionRef("streaming_vision")
    guide_agent: FunctionRef = FunctionRef("task_guide_agent")


@register_function(config_type=TaskGuideWorkflowConfig)
async def task_guide_workflow(config: TaskGuideWorkflowConfig, builder: Builder):
    state_group = await builder.get_function_group(config.task_state)
    state = await state_group.get_all_functions()
    get_status = state[f"{state_group.instance_name}__get_task_status"]
    control_group = await builder.get_function_group(config.task_control)
    controls = await control_group.get_all_functions()
    start_task = controls[f"{control_group.instance_name}__start_task"]
    reset_task = controls[f"{control_group.instance_name}__reset_task"]
    advance_task = controls[f"{control_group.instance_name}__advance_task"]
    vision = await builder.get_function(config.vision)
    guide_agent = await builder.get_function(config.guide_agent)

    async def observe_count(participant_id: str, step_id: str) -> FingerCount | None:
        visual = VisionResult.model_validate(
            await vision.ainvoke(
                VisionRequest(participant_id=participant_id, query=_COUNT_QUERY)
            )
        )
        count = parse_finger_count(visual.text)
        logger.info(
            "task vision completed pid={!r} step={} count={} hands={} confidence={}",
            participant_id,
            step_id,
            count.count if count else None,
            count.hands if count else None,
            count.confidence if count else None,
        )
        return count

    async def status(participant_id: str) -> TaskStatusResult:
        return TaskStatusResult.model_validate(await get_status.ainvoke({"participant_id": participant_id}))

    async def guide(request: TaskGuideRequest) -> TaskGuideReply:
        started = time.perf_counter()
        command = _command(request.text)
        control = _control_command(command)
        current = await status(request.participant_id)

        if control == "start":
            current = TaskStatusResult.model_validate(
                await start_task.ainvoke({"participant_id": request.participant_id})
            )
            response = format_task_status(current)
        elif control == "reset":
            current = TaskStatusResult.model_validate(
                await reset_task.ainvoke({"participant_id": request.participant_id})
            )
            response = format_task_status(current)
        elif control == "next":
            if current.progress.state != "running":
                response = format_task_status(current)
            else:
                current = TaskStatusResult.model_validate(
                    await advance_task.ainvoke({"participant_id": request.participant_id})
                )
                response = format_task_status(current)
        elif control == "status":
            response = format_task_status(current)
        elif control == "next_info":
            if current.next_step is None:
                response = "There is no later step; this is the final step."
            else:
                response = (
                    f"Next is {current.next_step.title}: {current.next_step.instructions} "
                    f"Current step remains {current.current_step.title}."
                )
        elif control == "validate":
            if current.progress.state != "running":
                response = format_task_status(current)
            else:
                assert current.current_step is not None
                expected = current.progress.step_index + 1
                count = await observe_count(request.participant_id, current.current_step.id)
                if count is None or count.confidence == "low":
                    response = f"{current.current_step.title} — I do not have a reliable finger count yet."
                elif count.count == expected:
                    fingers = "finger" if count.count == 1 else "fingers"
                    response = f"{current.current_step.title} — Yes, I see {count.count} extended {fingers}."
                else:
                    response = (
                        f"{current.current_step.title} — Not yet; I see {count.count}, "
                        f"but this step needs {expected}."
                    )
        elif not command:
            response = format_task_status(current)
        elif current.progress.state != "running":
            response = format_task_status(current)
        else:
            assert current.current_step is not None
            count = await observe_count(request.participant_id, current.current_step.id)
            if count is not None and any(marker in command for marker in _COUNT_QUESTION_MARKERS):
                response = f"{current.current_step.title} — {format_finger_count(count)}"
            else:
                reply = await guide_agent.ainvoke(
                    GuideAgentRequest(
                        participant_id=request.participant_id,
                        user_text=request.text,
                        latest_observation=(
                            format_finger_count(count) if count is not None else "The visual count was inconclusive."
                        ),
                    )
                )
                answer = str(
                    getattr(reply, "response", None)
                    or "I could not answer from the latest observation."
                )
                response = f"{current.current_step.title} — {answer}"

        logger.info(
            "task workflow completed pid={!r} command={!r} control={!r} "
            "state={} step={} elapsed_ms={:.0f} response={!r}",
            request.participant_id,
            command,
            control,
            current.progress.state,
            current.current_step.id if current.current_step else "complete",
            (time.perf_counter() - started) * 1_000,
            response,
        )
        return TaskGuideReply(response=response)

    yield FunctionInfo.from_fn(
        guide,
        description="Control a hand-counting task or answer from its latest monitored observation.",
    )


__all__ = ["TaskGuideWorkflowConfig", "format_task_status"]
