# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-calling LLM loops for workflow state updates and user answers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from xr_ai_models import ChatMessage, LLMService, ToolDef

from .tools import AgentTool, GuideTools, ToolCatalog
from .workflow import (
    WorkflowDefinition,
    WorkflowSession,
    WorkflowStep,
    context_for_prompt,
)

_OBSERVATION_LOG_TOOL = "get_recent_vlm_observations"
_VISUAL_INSPECTION_TOOL = "inspect_current_view"
_NEXT_STEP_TOOL = "get_next_workflow_step"
_TASK_CONTROL_KEYS = {
    "navigation_examples",
    "start_triggers",
    "status_triggers",
    "stop_triggers",
}
VisualQueryFn = Callable[[str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class StepAgentResult:
    """Structured output from one step-agent iteration."""

    context_patch: dict[str, Any]
    step_state: str = "started"
    ready_to_advance: bool = False
    assistant_message: str = ""
    speak: bool = False


@dataclass(frozen=True, slots=True)
class NavigationIntent:
    """User utterance classification for workflow-level commands."""

    intent: str = "answer"
    skip_requested: bool = False
    explicit_command: bool = False
    confidence: float = 0.0


class WorkflowAgent:
    """Runs the generic agentic loop around YAML-authored step prompts."""

    def __init__(
        self,
        *,
        llm: LLMService,
        tools: GuideTools,
        workflow: WorkflowDefinition,
        answer_prompt: Path,
    ) -> None:
        self._llm = llm
        self._workflow = workflow
        self._tool_catalog = ToolCatalog(_agent_tools(tools))
        self._answer_prompt_path = answer_prompt
        self._answer_prompt_cache = answer_prompt.read_text(encoding="utf-8").strip()

    async def run_step(
        self,
        *,
        step: WorkflowStep,
        participant_id: str,
        context: dict[str, Any],
        observation_log: list[dict[str, Any]],
        vlm_observation: str,
    ) -> StepAgentResult:
        prompt_context = self._workflow.context_for_step(
            step,
            context,
        )
        messages = [
            ChatMessage(role="system", content=_STEP_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=self._step_user_prompt(step, prompt_context, vlm_observation),
            ),
        ]
        raw = await self._run_tool_loop(
            messages,
            max_tokens=1536,
            tools=self._step_tools(step),
        )
        logger.debug(
            "step agent raw pid={} step={} text={!r}",
            participant_id,
            step.id,
            raw[:1000],
        )
        obj = _json_object(raw)
        if not isinstance(obj, dict):
            logger.warning("step agent did not return JSON: {!r}", raw[:200])
            return StepAgentResult(context_patch={})

        context_patch = obj.get("context")
        if not isinstance(context_patch, dict):
            valid = step.writable_fields
            context_patch = {key: value for key, value in obj.items() if key in valid}
        return StepAgentResult(
            context_patch=dict(context_patch),
            step_state=_valid_step_state(obj.get("step_state")),
            ready_to_advance=bool(obj.get("ready_to_advance", False)),
            assistant_message=str(obj.get("assistant_message") or "").strip(),
            speak=bool(obj.get("speak", False)),
        )

    async def classify_intent(
        self,
        *,
        transcript: str,
        session: WorkflowSession | None,
        current_step: WorkflowStep,
        recent_turns: list[tuple[str, str]],
    ) -> NavigationIntent:
        full_context = session.context if session is not None else self._workflow.initial_context()
        context = self._workflow.context_for_step(current_step, full_context)
        prior_user_request = recent_turns[-1][0] if recent_turns else "(none)"
        messages = [
            ChatMessage(role="system", content=_NAVIGATION_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"[Task]\n{json.dumps(self._workflow.task, ensure_ascii=True)}\n\n"
                    f"[Workflow active]\n{session is not None and session.active}\n\n"
                    f"[Current step]\n"
                    f"id={current_step.id}\n"
                    f"name={current_step.name}\n"
                    f"description={current_step.description}\n\n"
                    f"[Step state]\n"
                    f"state={session.step_state if session is not None else 'idle'}\n"
                    f"ready={bool(session and session.ready_step_id == current_step.id)}\n"
                    f"workflow_active={bool(session and session.active)}\n\n"
                    f"[Workflow context]\n{context_for_prompt(context)}\n\n"
                    f"[Prior user request for follow-up reference]\n"
                    f"{prior_user_request}\n"
                    f"Do not reuse any earlier answer or measured value.\n\n"
                    f"[User utterance]\n{transcript}"
                ),
            ),
        ]
        try:
            response = await self._llm.chat(
                messages,
                max_tokens=64,
                temperature=0.0,
                enable_thinking=False,
                timeout=self._workflow.navigation_timeout_s,
            )
        except Exception:
            logger.exception("guide navigation classifier failed")
            return NavigationIntent()

        logger.debug(
            "navigation classifier raw step={} text={!r} response={!r}",
            current_step.id,
            transcript,
            (response.content or "")[:500],
        )
        obj = _json_object(response.content or "")
        if not isinstance(obj, dict):
            return NavigationIntent()
        intent = str(obj.get("intent") or "answer").casefold().strip()
        if intent not in {"start", "stop", "status", "advance", "answer"}:
            intent = "answer"
        confidence = _as_float(obj.get("confidence"), default=0.0)
        return NavigationIntent(
            intent=intent,
            skip_requested=bool(obj.get("skip_requested", False)),
            explicit_command=bool(obj.get("explicit_command", False)),
            confidence=max(0.0, min(confidence, 1.0)),
        )

    async def answer_user(
        self,
        *,
        transcript: str,
        session: WorkflowSession | None,
        current_step: WorkflowStep,
        observation_log: list[dict[str, Any]],
        recent_turns: list[tuple[str, str]],
        visual_query: VisualQueryFn | None = None,
    ) -> str:
        context = session.context if session is not None else self._workflow.initial_context()
        return await self.answer_step(
            transcript=transcript,
            context=context,
            current_step=current_step,
            step_state=session.step_state if session is not None else "idle",
            ready=bool(session and session.ready_step_id == current_step.id),
            workflow_active=bool(session and session.active),
            observation_log=observation_log,
            recent_turns=recent_turns,
            visual_query=visual_query,
        )

    async def answer_step(
        self,
        *,
        transcript: str,
        context: dict[str, Any],
        current_step: WorkflowStep,
        step_state: str,
        ready: bool,
        workflow_active: bool,
        observation_log: list[dict[str, Any]],
        recent_turns: list[tuple[str, str]],
        visual_query: VisualQueryFn | None = None,
    ) -> str:
        """Answer one read-only voice event in the current step's scope."""

        system = self._read_answer_prompt()
        prompt_context = self._workflow.context_for_step(current_step, context)
        prior_user_request = recent_turns[-1][0] if recent_turns else "(none)"
        latest_observation = _latest_step_observation(
            observation_log,
            current_step.id,
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(
                role="user",
                content=(
                    f"[Task]\n{json.dumps(_task_context(self._workflow.task), ensure_ascii=True)}\n\n"
                    f"[Current step]\n"
                    f"id={current_step.id}\n"
                    f"name={current_step.name}\n"
                    f"description={current_step.description}\n\n"
                    f"[Step procedure]\n{current_step.agent_prompt}\n\n"
                    f"[Prior user request for follow-up reference]\n"
                    f"{prior_user_request}\n"
                    f"Do not reuse any earlier answer or measured value.\n\n"
                    f"[Workflow snapshot]\n"
                    f"The current-step context and latest step-monitor observation below "
                    f"supersede older workflow observations, but they are not a fresh "
                    f"camera inspection for the current request.\n\n"
                    f"[Step state]\n"
                    f"state={step_state}\n"
                    f"ready={ready}\n"
                    f"workflow_active={workflow_active}\n\n"
                    f"[Current-step context only]\n{context_for_prompt(prompt_context)}\n\n"
                    f"[Latest step-monitor observation]\n{latest_observation}\n\n"
                    f"[VLM observation log]\n"
                    f"A rolling internal log is available through "
                    f"{_OBSERVATION_LOG_TOOL}. Use it only when visual evidence "
                    f"is needed to answer the wearer.\n\n"
                    f"[User request]\n{transcript}"
                ),
            ),
        ]
        response = await self._run_tool_loop(
            messages,
            max_tokens=256,
            tools=self._answer_tools(
                current_step=current_step,
                observation_log=observation_log,
                visual_query=visual_query,
            ),
        )
        logger.info(
            "answer agent response active={} step={} text={!r}",
            workflow_active,
            current_step.id,
            response[:1000],
        )
        return response.strip() or "I could not determine that from the available information."

    def _step_user_prompt(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        vlm_observation: str,
    ) -> str:
        schema = {
            "context": step.context_schema(),
            "ready_to_advance": {
                "type": "boolean",
                "description": "True only when this step's completion condition is satisfied.",
            },
            "step_state": {
                "type": "string",
                "enum": ["started", "needs_input", "complete"],
                "description": "Mini state for this step, not the workflow step number.",
            },
            "assistant_message": {
                "type": "string",
                "description": ("Empty unless an immediate visible safety correction is required."),
            },
            "speak": {
                "type": "boolean",
                "description": ("True only for an immediate visible safety correction."),
            },
        }
        return (
            f"[Task]\n{json.dumps(_task_context(self._workflow.task), ensure_ascii=True)}\n\n"
            f"[Step]\n"
            f"id={step.id}\n"
            f"name={step.name}\n"
            f"description={step.description}\n\n"
            f"[Current context]\n{context_for_prompt(context)}\n\n"
            f"[Latest VLM observation]\n{vlm_observation or '(empty)'}\n\n"
            f"[Step-specific procedure]\n{step.agent_prompt}\n\n"
            f"[Advance condition]\n{json.dumps(step.advance_when, ensure_ascii=True)}\n\n"
            f"[Required final JSON shape]\n{json.dumps(schema, indent=2, ensure_ascii=True)}"
        )

    async def _run_tool_loop(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        tools: list[AgentTool] | None = None,
    ) -> str:
        available_tools = tools or []
        definitions = [tool.definition for tool in available_tools]
        catalog = ToolCatalog(available_tools)
        for iteration in range(self._workflow.max_agent_iterations):
            try:
                response = await self._llm.chat(
                    messages,
                    tools=definitions,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    enable_thinking=False,
                )
            except Exception:
                logger.exception("guide agent LLM call failed")
                return ""

            content = (response.content or "").strip()
            tool_calls = response.tool_calls or ()
            if not tool_calls:
                return content

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=list(tool_calls),
                )
            )
            for call in tool_calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                logger.debug(
                    "guide tool call iter={} tool={} args={}",
                    iteration,
                    call.name,
                    arguments,
                )
                try:
                    result = await catalog.invoke(call.name, arguments)
                except Exception as exc:
                    logger.exception("guide tool failed: {}", call.name)
                    result = {"error": str(exc)}
                logger.debug(
                    "guide tool result iter={} tool={} result={}",
                    iteration,
                    call.name,
                    result,
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=True, default=str),
                        tool_call_id=call.id,
                    )
                )
        return ""

    def _step_tools(
        self,
        step: WorkflowStep,
    ) -> list[AgentTool]:
        if not step.agent_tools:
            return []
        missing = self._tool_catalog.missing(step.agent_tools)
        for name in sorted(missing):
            logger.warning("step {} references unknown agent tool {}", step.id, name)
        enabled = self._tool_catalog.select(step.agent_tools)
        logger.debug(
            "step agent tools step={} enabled={}",
            step.id,
            [tool.name for tool in enabled],
        )
        return enabled

    def _answer_tools(
        self,
        *,
        current_step: WorkflowStep,
        observation_log: list[dict[str, Any]],
        visual_query: VisualQueryFn | None = None,
    ) -> list[AgentTool]:
        async def recent(arguments: dict[str, Any]) -> dict[str, Any]:
            return _recent_observations(observation_log, arguments)

        async def inspect(arguments: dict[str, Any]) -> dict[str, Any]:
            question = str(arguments.get("question") or "").strip()
            if visual_query is None:
                return {"error": "A live camera view is not available."}
            if not question:
                return {"error": "question is required"}
            return await visual_query(question)

        async def next_step(_arguments: dict[str, Any]) -> dict[str, Any]:
            following = (
                self._workflow.first_active_step()
                if current_step.is_idle
                else self._workflow.next_step(current_step.id)
            )
            if following is None:
                return {"has_next_step": False}
            return {
                "has_next_step": True,
                "id": following.id,
                "name": following.name,
                "description": following.description,
                "on_enter_message": following.on_enter_message,
            }

        scoped = [
            AgentTool(_observation_log_tool_def(), recent, read_only=True),
            AgentTool(_visual_inspection_tool_def(), inspect, read_only=True),
            AgentTool(_next_step_tool_def(), next_step, read_only=True),
        ]
        return [
            *self._tool_catalog.select(read_only_only=True),
            *scoped,
        ]

    def _read_answer_prompt(self) -> str:
        try:
            text = self._answer_prompt_path.read_text(encoding="utf-8").strip()
            self._answer_prompt_cache = text
            return text
        except OSError:
            logger.warning("answer prompt unreadable; using cached prompt")
            return self._answer_prompt_cache


_STEP_SYSTEM_PROMPT = """You run one iteration of a YAML-defined workflow step.
You receive the latest optional VLM caption, current workflow context, and the
step-specific procedure. Interpret the caption and perform every state update
through the returned context patch. Call the step's enabled tools whenever its
procedure requires current time, timer status, RAG, or other external data.

Stay inside this step. Do not discuss, prepare, summarize, or give instructions
for any future step. The latest VLM caption is the only authority for fields that
describe what is visible now. Current context is prior state, not current visual
evidence. When the procedure declares a field mutable, update it from every new
caption and never substitute an older value. Derive readiness only from the
latest caption, the projected context, enabled tool results, and the explicit
advance condition.

Return only valid JSON. The top-level object must contain:
- context: a partial patch containing only fields this step can write and only
  values supported by the latest VLM observation, RAG, tools, or context.
- ready_to_advance: whether this step is complete.
- step_state: started, needs_input, or complete for this step's internal state.
- assistant_message: optional concise high-level guidance or a missing-info
  request. Leave it empty unless the latest evidence shows an immediate safety
  problem requiring a correction.
- speak: true only for that immediate safety correction; otherwise false. The
  worker owns entry instructions, reminders, completion, and navigation speech.

Never copy, summarize, or narrate the VLM observation as assistant_message.
The VLM observation is internal evidence. assistant_message is only for a short
urgent safety correction. Do not announce status, readiness, tea details, the
current step, or any next action from this loop.

Do not change the workflow step number. Do not invent visible facts. A writable
observation may change more than once within one step; prefer newer evidence to
older context. For durable milestones, retain established true values unless
the procedure says otherwise. Omit unchanged fields instead of copying the
entire context."""


_NAVIGATION_SYSTEM_PROMPT = """Classify one user utterance for a guided workflow.
Return only valid JSON with:
- intent: one of start, stop, status, advance, answer
- skip_requested: true only when the wearer wants to skip or proceed before the
  current step is known complete
- explicit_command: true only when the wearer directly commands the workflow to
  start, stop, or move; false for observations, completion reports, and questions
- confidence: number from 0 to 1

Use advance when the wearer is commanding movement to the next step, including
uncommon or conversational wording with that meaning. A report such as "I put
the tea in the cup" describes task progress but does not command workflow
movement, so classify it as answer with explicit_command false.
Use answer when the wearer is asking a question, including "what is the next
step?", "can I proceed?", "what should I do?", or task-specific questions.
Use status only for requests asking where the workflow is. Use stop for cancel
or end guidance requests. Use start for requests to begin the task when idle."""


def _observation_log_tool_def() -> ToolDef:
    return ToolDef(
        name=_OBSERVATION_LOG_TOOL,
        description=(
            "Return recent internal VLM observations for this active workflow session. "
            "Use this only when the wearer asks what was observed, asks about visual "
            "status, or previous visual evidence is needed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of recent observations to return, from 1 to 20.",
                    "default": 6,
                },
                "step_id": {
                    "type": "integer",
                    "description": "Optional workflow step ID to filter observations.",
                },
            },
            "additionalProperties": False,
        },
    )


def _agent_tools(tools: GuideTools) -> list[AgentTool]:
    """Adapt old tool providers while preferring capability-aware providers."""

    provider = getattr(tools, "agent_tools", None)
    if callable(provider):
        return list(provider())

    adapted: list[AgentTool] = []
    for definition in tools.definitions():

        async def invoke(
            arguments: dict[str, Any],
            *,
            name: str = definition.name,
        ) -> dict[str, Any]:
            return await tools.invoke(name, arguments)

        adapted.append(AgentTool(definition, invoke, read_only=False))
    return adapted


def _task_context(task: dict[str, Any]) -> dict[str, Any]:
    """Remove navigation configuration from non-navigation model prompts."""

    return {key: value for key, value in task.items() if key not in _TASK_CONTROL_KEYS}


def _visual_inspection_tool_def() -> ToolDef:
    return ToolDef(
        name=_VISUAL_INSPECTION_TOOL,
        description=(
            "Capture a fresh camera frame and answer a question about what is visible "
            "right now. Use this for present-tense visual questions, object or text "
            "identification, current readings, quantities, appearance, placement, or "
            "safety checks. Do not rely on an older workflow caption when this tool can "
            "inspect the current view."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The wearer's complete visual question, preserving the object, "
                        "property, or text they want inspected."
                    ),
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    )


def _next_step_tool_def() -> ToolDef:
    return ToolDef(
        name=_NEXT_STEP_TOOL,
        description=(
            "Return the exact next YAML-defined workflow step. Use only when the "
            "wearer explicitly asks what comes next; never call it for an ordinary "
            "current-step question or progress report."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


def _recent_observations(
    observation_log: list[dict[str, Any]],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        count = int(arguments.get("count") or 6)
    except (TypeError, ValueError):
        count = 6
    count = max(1, min(count, 20))

    raw_step_id = arguments.get("step_id")
    try:
        step_id = int(raw_step_id) if raw_step_id is not None else None
    except (TypeError, ValueError):
        step_id = None

    entries = [dict(entry) for entry in observation_log if step_id is None or entry.get("step_id") == step_id]
    return {
        "entries": entries[-count:],
        "returned": min(len(entries), count),
        "available": len(entries),
    }


def _latest_step_observation(
    observation_log: list[dict[str, Any]],
    step_id: int,
) -> str:
    latest = next(
        (
            entry
            for entry in reversed(observation_log)
            if entry.get("step_id") == step_id and entry.get("kind", "step_monitor") == "step_monitor"
        ),
        None,
    )
    if latest is None:
        return "(none)"
    return json.dumps(latest, ensure_ascii=True, sort_keys=True)


def _valid_step_state(value: Any) -> str:
    state = str(value or "").casefold().strip()
    return state if state in {"started", "needs_input", "complete"} else "started"


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_object(text: str) -> dict[str, Any] | None:
    extracted = _extract_json(text)
    if extracted is None:
        return None
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_json(text: str) -> str | None:
    depth, start, in_string, escape = 0, -1, False, False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : index + 1]
    return None


__all__ = ["NavigationIntent", "StepAgentResult", "WorkflowAgent"]
