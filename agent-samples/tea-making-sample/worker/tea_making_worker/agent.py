# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-calling LLM loops for workflow state updates and user answers."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from xr_ai_models import ChatMessage, LLMService, ToolCall, ToolDef

from .tools import GuideTools
from .workflow import (
    WorkflowDefinition,
    WorkflowSession,
    WorkflowStep,
    context_for_prompt,
)

_OBSERVATION_LOG_TOOL = "get_recent_vlm_observations"
_VISUAL_INSPECTION_TOOL = "inspect_current_view"
VisualQueryFn = Callable[[str], Awaitable[dict[str, Any]]]
_VLM_VERDICT_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9_ ]+)\s*:\s*(yes|no|unclear)\b",
    re.IGNORECASE | re.MULTILINE,
)


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
        self._tools = tools
        self._workflow = workflow
        self._tool_defs = [
            *tools.definitions(),
            _observation_log_tool_def(),
            _visual_inspection_tool_def(),
        ]
        self._answer_prompt_path = answer_prompt
        self._answer_prompt_cache = answer_prompt.read_text(encoding="utf-8").strip()

    async def run_step(
        self,
        *,
        step: WorkflowStep,
        session: WorkflowSession,
        vlm_observation: str,
    ) -> StepAgentResult:
        automatic_patch = await self._auto_invoke_step_tools(
            step,
            vlm_observation,
            session.context,
        )
        prompt_context = self._workflow.context_for_step(
            step,
            {**session.context, **automatic_patch},
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
            observation_log=session.observation_log,
            tool_defs=self._step_tool_defs(step, vlm_observation),
        )
        logger.debug(
            "step agent raw pid={} step={} text={!r}",
            session.participant_id,
            step.id,
            raw[:1000],
        )
        obj = _json_object(raw)
        if not isinstance(obj, dict):
            logger.warning("step agent did not return JSON: {!r}", raw[:200])
            return StepAgentResult(context_patch=automatic_patch)

        context_patch = obj.get("context")
        if not isinstance(context_patch, dict):
            valid = step.writable_fields
            context_patch = {key: value for key, value in obj.items() if key in valid}
        context_patch = {**context_patch, **automatic_patch}

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
        history = (
            "\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in recent_turns[-2:]) or "(none)"
        )
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
                    f"[Recent conversation]\n{history}\n\n"
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
        system = self._read_answer_prompt()
        context = session.context if session is not None else self._workflow.initial_context()
        timer_state = self._timer_state_for_prompt(session)
        history = (
            "\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in recent_turns[-4:]) or "(none)"
        )
        latest_observation = _latest_step_observation(
            observation_log,
            current_step.id,
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(
                role="user",
                content=(
                    f"[Task]\n{json.dumps(self._workflow.task, ensure_ascii=True)}\n\n"
                    f"[Current step]\n"
                    f"id={current_step.id}\n"
                    f"name={current_step.name}\n"
                    f"description={current_step.description}\n\n"
                    f"[Step procedure]\n{current_step.agent_prompt}\n\n"
                    f"[Recent conversation]\n{history}\n\n"
                    f"[Authoritative current state]\n"
                    f"The workflow context and latest VLM observation below supersede "
                    f"older conversation turns and older observations.\n\n"
                    f"[Step state]\n"
                    f"state={session.step_state if session is not None else 'idle'}\n"
                    f"ready={bool(session and session.ready_step_id == current_step.id)}\n"
                    f"workflow_active={bool(session and session.active)}\n\n"
                    f"[Workflow context]\n{context_for_prompt(context)}\n\n"
                    f"[Authoritative timer state]\n{timer_state}\n\n"
                    f"[Latest VLM observation]\n{latest_observation}\n\n"
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
            max_tokens=1024,
            observation_log=observation_log,
            visual_query=visual_query,
        )
        logger.info(
            "answer agent response active={} step={} text={!r}",
            session is not None and session.active,
            current_step.id,
            response[:1000],
        )
        return response.strip() or "Done."

    def _timer_state_for_prompt(self, session: WorkflowSession | None) -> str:
        if session is None:
            return "No active timer."
        status = self._workflow.find_timer_status(
            session.context,
            current_step_id=session.step_id,
            now_us=time.time_ns() // 1_000,
        )
        if status is None:
            return "No timer start and duration are currently recorded."
        return json.dumps(
            {
                "label": status.label,
                "started_at_us": status.started_at_us,
                "duration_seconds": status.duration_seconds,
                "elapsed_seconds": status.elapsed_seconds,
                "remaining_seconds": status.remaining_seconds,
                "expired": status.expired,
                "instruction": (
                    "Treat these derived values as authoritative. Never claim the timer "
                    "is complete when expired is false."
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
        )

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
                "description": ("Optional short guidance, correction, or missing-info request."),
            },
            "speak": {
                "type": "boolean",
                "description": ("Reserved for urgent notices; false for ordinary missing information."),
            },
        }
        return (
            f"[Task]\n{json.dumps(self._workflow.task, ensure_ascii=True)}\n\n"
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
        observation_log: list[dict[str, Any]] | None = None,
        tool_defs: list[ToolDef] | None = None,
        visual_query: VisualQueryFn | None = None,
    ) -> str:
        available_tools = self._tool_defs if tool_defs is None else tool_defs
        for iteration in range(self._workflow.max_agent_iterations):
            try:
                response = await self._llm.chat(
                    messages,
                    tools=available_tools,
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
                result = await self._execute_tool(
                    call,
                    iteration=iteration,
                    observation_log=observation_log or [],
                    visual_query=visual_query,
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=True, default=str),
                        tool_call_id=call.id,
                    )
                )
        return ""

    def _step_tool_defs(
        self,
        step: WorkflowStep,
        vlm_observation: str,
    ) -> list[ToolDef]:
        if not step.agent_tools:
            return []
        definitions = {tool.name: tool for tool in self._tool_defs}
        enabled: list[ToolDef] = []
        for name, policy in step.agent_tools.items():
            tool = definitions.get(name)
            if tool is None:
                logger.warning("step {} references unknown agent tool {}", step.id, name)
                continue
            if _tool_policy_met(policy, vlm_observation):
                enabled.append(tool)
        logger.debug(
            "step agent tools step={} enabled={}",
            step.id,
            [tool.name for tool in enabled],
        )
        return enabled

    async def _auto_invoke_step_tools(
        self,
        step: WorkflowStep,
        vlm_observation: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        for name, policy in step.agent_tools.items():
            if not policy.get("auto_invoke") or not _tool_policy_met(
                policy,
                vlm_observation,
            ):
                continue
            empty_field = str(policy.get("when_context_empty") or "").strip()
            effective_context = {**context, **patch}
            if empty_field and _context_value_present(_dotted_value(effective_context, empty_field)):
                logger.debug(
                    "automatic step tool skipped step={} tool={} field={} reason=present",
                    step.id,
                    name,
                    empty_field,
                )
                continue
            arguments = policy.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            logger.info("automatic step tool call step={} tool={}", step.id, name)
            try:
                result = await self._tools.invoke(name, arguments)
            except Exception:
                logger.exception("automatic step tool failed step={} tool={}", step.id, name)
                continue
            logger.debug(
                "automatic step tool result step={} tool={} result={}",
                step.id,
                name,
                result,
            )
            outputs = policy.get("context_outputs") or {}
            if not isinstance(result, dict) or not isinstance(outputs, dict):
                logger.warning(
                    "automatic step tool returned unmappable result step={} tool={}",
                    step.id,
                    name,
                )
                continue
            mapped = {
                str(context_field): _dotted_value(result, str(result_field))
                for result_field, context_field in outputs.items()
            }
            mapped = {key: value for key, value in mapped.items() if value is not None}
            patch.update(mapped)
            logger.info(
                "automatic step tool captured step={} tool={} context={}",
                step.id,
                name,
                mapped,
            )
        return patch

    async def _execute_tool(
        self,
        call: ToolCall,
        *,
        iteration: int,
        observation_log: list[dict[str, Any]],
        visual_query: VisualQueryFn | None = None,
    ) -> dict[str, Any]:
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        logger.debug("guide tool call iter={} tool={} args={}", iteration, call.name, arguments)
        if call.name == _OBSERVATION_LOG_TOOL:
            result = _recent_observations(observation_log, arguments)
            logger.debug(
                "guide tool result iter={} tool={} result={}",
                iteration,
                call.name,
                result,
            )
            return result
        if call.name == _VISUAL_INSPECTION_TOOL:
            question = str(arguments.get("question") or "").strip()
            if visual_query is None:
                return {"error": "A live camera view is not available."}
            if not question:
                return {"error": "question is required"}
            try:
                result = await visual_query(question)
                logger.debug(
                    "guide tool result iter={} tool={} result={}",
                    iteration,
                    call.name,
                    result,
                )
                return result
            except Exception as exc:
                logger.exception("live visual inspection failed")
                return {"error": str(exc)}
        try:
            result = await self._tools.invoke(call.name, arguments)
            logger.debug(
                "guide tool result iter={} tool={} result={}",
                iteration,
                call.name,
                result,
            )
            return result
        except Exception as exc:
            logger.exception("guide tool failed: {}", call.name)
            return {"error": str(exc)}

    def _read_answer_prompt(self) -> str:
        try:
            text = self._answer_prompt_path.read_text(encoding="utf-8").strip()
            self._answer_prompt_cache = text
            return text
        except OSError:
            logger.warning("answer prompt unreadable; using cached prompt")
            return self._answer_prompt_cache


_STEP_SYSTEM_PROMPT = """You run one step of a YAML-defined guided workflow.
You receive a VLM observation produced by the step's VLM prompt, the current
workflow context, and the step-specific procedure. Use tools only when needed:
RAG for task reference information, current time for timestamps, and the VLM
observation log if earlier visual evidence matters.

Return only valid JSON. The top-level object must contain:
- context: a partial patch containing only fields this step can write and only
  values supported by the latest VLM observation, RAG, tools, or context.
- ready_to_advance: whether this step is complete.
- step_state: started, needs_input, or complete for this step's internal state.
- assistant_message: optional concise high-level guidance or a missing-info
  request. The worker controls reminder timing, so do not repeat yourself.
- speak: false for ordinary missing information. The worker controls when to
  speak delayed reminders and ready notices.

Never copy, summarize, or narrate the VLM observation as assistant_message.
The VLM observation is internal evidence. assistant_message is only for a short
actionable correction, missing-info request, or useful "ready for next" style
notice.

Do not change the workflow step number. Do not invent visible facts. A writable
observation may change more than once within one step; prefer newer evidence to
older context. Omit unchanged fields instead of copying the entire context."""


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
        (entry for entry in reversed(observation_log) if entry.get("step_id") == step_id),
        None,
    )
    if latest is None:
        return "(none)"
    return json.dumps(latest, ensure_ascii=True, sort_keys=True)


def _tool_policy_met(policy: dict[str, Any], vlm_observation: str) -> bool:
    verdict_name = str(policy.get("vlm_verdict") or "").strip()
    if not verdict_name:
        return True
    raw_expected = policy.get("equals", "yes")
    if isinstance(raw_expected, bool):
        expected = "yes" if raw_expected else "no"
    else:
        expected = str(raw_expected).casefold().strip()
    verdicts = {
        _normalize_verdict_name(match.group(1)): match.group(2).casefold()
        for match in _VLM_VERDICT_RE.finditer(vlm_observation)
    }
    return verdicts.get(_normalize_verdict_name(verdict_name)) == expected


def _normalize_verdict_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def _dotted_value(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _context_value_present(value: Any) -> bool:
    return value not in (None, "", 0, 0.0, False, [], {})


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
