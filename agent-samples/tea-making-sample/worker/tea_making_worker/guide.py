# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-facing guided workflow orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from .agent import NavigationIntent, WorkflowAgent
from .step_mechanism import CaptionAgentStepMechanism, StepEvent, StepMechanisms
from .vision import FrameUnavailable, StepVision
from .workflow import (
    WorkflowDefinition,
    WorkflowSession,
    WorkflowStep,
    render_template,
    speech_text,
)

NoticeFn = Callable[[str, str], Awaitable[None]]


class WorkflowGuide:
    """Owns per-participant sessions for a YAML-defined guided workflow."""

    def __init__(
        self,
        *,
        workflow: WorkflowDefinition,
        vision: StepVision,
        agent: WorkflowAgent,
        notice: NoticeFn,
        step_mechanisms: StepMechanisms | None = None,
    ) -> None:
        self._workflow = workflow
        self._vision = vision
        self._agent = agent
        self._notice = notice
        self._step_mechanisms = step_mechanisms or StepMechanisms(
            [
                CaptionAgentStepMechanism(
                    workflow=workflow,
                    vision=vision,
                    agent=agent,
                )
            ]
        )
        self._step_mechanisms.validate(workflow)
        self._sessions: dict[str, WorkflowSession] = {}
        self._history: dict[str, list[tuple[str, str]]] = {}
        self._closed = asyncio.Event()

    async def handle_query(
        self,
        *,
        participant_id: str,
        text: str,
    ) -> str:
        clean = text.strip()
        if not clean:
            return ""

        session = self._sessions.get(participant_id)
        if session is not None:
            session.connected = True
        step = self._current_step(session)
        logger.info(
            "guide query pid={} active={} step={} state={} text={!r}",
            participant_id,
            bool(session and session.active),
            step.id,
            session.step_state if session is not None else "idle",
            clean,
        )
        if session is not None and session.active:
            session.user_turn_active = True
        response = ""
        deferred_notice = ""
        intent = NavigationIntent()
        try:
            response = _idle_navigation_answer(clean, session, self._workflow)
            if not response:
                intent = _local_intent(
                    clean,
                    self._workflow,
                    active=session is not None and session.active,
                )
                if intent is not None:
                    logger.debug(
                        "guide local intent pid={} intent={} confidence={}",
                        participant_id,
                        intent.intent,
                        intent.confidence,
                    )
                else:
                    classified = await self._agent.classify_intent(
                        transcript=clean,
                        session=session,
                        current_step=step,
                        recent_turns=self._history.get(participant_id, []),
                    )
                    intent = _validated_model_intent(
                        clean,
                        classified,
                        self._workflow,
                        active=bool(session and session.active),
                    )
                    logger.info(
                        "guide llm intent pid={} proposed={} accepted={} skip={} confidence={}",
                        participant_id,
                        classified.intent,
                        intent.intent,
                        intent.skip_requested,
                        classified.confidence,
                    )

            if response:
                logger.info("guide deterministic answer pid={} text={!r}", participant_id, response)
            elif intent.intent == "stop":
                response = await self.stop(participant_id)
            elif intent.intent == "start":
                response = await self.start(participant_id)
            elif intent.intent == "status":
                response = self.status(participant_id)
            elif intent.intent == "advance":
                response = await self.advance(
                    participant_id,
                    manual=intent.skip_requested,
                )
            else:
                logger.debug(
                    "guide voice event dispatch pid={} step={} mechanism={}",
                    participant_id,
                    step.id,
                    step.mechanism,
                )
                response = await self._handle_voice_event(
                    participant_id=participant_id,
                    transcript=clean,
                    session=session,
                    step=step,
                )
        finally:
            if session is not None:
                session.user_turn_active = False
                deferred_notice = session.deferred_notice
                session.deferred_notice = ""

        response = speech_text(response)
        if intent.intent == "answer":
            self._record(participant_id, clean, response)
        elif intent.intent in {"start", "advance", "stop"}:
            self._history.pop(participant_id, None)
        logger.info(
            "guide response pid={} intent={} text={!r}",
            participant_id,
            intent.intent,
            response,
        )
        if deferred_notice and response:
            logger.info(
                "guide deferred notice pid={} text={!r}",
                participant_id,
                deferred_notice,
            )
            await self._notice(participant_id, deferred_notice)
        return response

    async def start(self, participant_id: str) -> str:
        step = self._workflow.first_active_step()
        self._history.pop(participant_id, None)
        session = WorkflowSession(
            participant_id=participant_id,
            step_id=step.id,
            context=self._workflow.initial_context(),
        )
        self._begin_step(session, step)
        self._sessions[participant_id] = session
        message = self._message(
            step.on_enter_message,
            session=session,
            step=step,
            fallback=f"Step {step.id}: {step.name}.",
        )
        logger.info("guide start pid={} step={} message={!r}", participant_id, step.id, message)
        return message

    async def stop(self, participant_id: str) -> str:
        session = self._sessions.get(participant_id)
        if session is not None:
            self._set_idle(session)
        logger.info("guide stop pid={}", participant_id)
        return "Guidance stopped."

    async def advance(self, participant_id: str, *, manual: bool = False) -> str:
        session = self._sessions.get(participant_id)
        if session is None or not session.active:
            return "No guided workflow is active."
        async with session.lock:
            current = self._workflow.step_by_id(session.step_id)
            complete = session.ready_step_id == current.id or self._workflow.advance_when_met(current, session.context)
            skipped = not complete
            applied: dict[str, Any] = {}
            if skipped:
                applied = self._workflow.apply_skip_defaults(current, session.context)
                logger.info(
                    "guide advance applying skip defaults pid={} step={} fields={}",
                    participant_id,
                    current.id,
                    sorted(applied),
                )
            next_step = self._workflow.next_step(current.id)
            if next_step is None:
                template = current.on_skip_message if skipped else current.on_complete_message
                message = self._message(
                    template,
                    session=session,
                    step=current,
                    fallback=f"{self._workflow.task.get('name', 'Workflow')} is complete.",
                )
                logger.info(
                    "guide complete pid={} step={} skipped={} message={!r}",
                    participant_id,
                    current.id,
                    skipped or manual,
                    message,
                )
                self._set_idle(session)
                return _advance_prefix(skipped or manual) + message

            self._begin_step(session, next_step)
            message = self._message(
                next_step.on_enter_message,
                session=session,
                step=next_step,
                fallback=f"Step {next_step.id}: {next_step.name}.",
            )
            logger.info(
                "guide advance pid={} from_step={} to_step={} skipped={} message={!r}",
                participant_id,
                current.id,
                next_step.id,
                skipped or manual,
                message,
            )
            return _advance_prefix(skipped or manual) + message

    def status(self, participant_id: str) -> str:
        session = self._sessions.get(participant_id)
        if session is None or not session.active:
            return "No guided workflow is active."
        step = self._workflow.step_by_id(session.step_id)
        suffix = " It is ready; say next when you want to continue." if session.ready_step_id == step.id else ""
        return f"On step {step.id}: {step.name}. State: {session.step_state}.{suffix}"

    async def monitor_forever(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(self._workflow.monitor_interval_s)
            sessions = list(self._sessions.values())
            results = await asyncio.gather(
                *(self._evaluate(session) for session in sessions),
                return_exceptions=True,
            )
            for session, result in zip(sessions, results, strict=True):
                if isinstance(result, BaseException):
                    logger.opt(
                        exception=(type(result), result, result.__traceback__),
                    ).error(
                        "guide monitor failed pid={} step={}",
                        session.participant_id,
                        session.step_id,
                    )

    async def release(self, participant_id: str) -> None:
        self._vision.release(participant_id)
        session = self._sessions.pop(participant_id, None)
        self._history.pop(participant_id, None)
        if session is not None:
            previous_step = session.step_id
            self._set_idle(session)
            session.connected = False
            logger.info(
                "guide participant disconnected and cleared pid={} previous_step={}",
                participant_id,
                previous_step,
            )

    async def reset(self, participant_id: str) -> None:
        """Discard any stale participant state before a new connection."""

        session = self._sessions.pop(participant_id, None)
        self._history.pop(participant_id, None)
        if session is not None:
            self._set_idle(session)
        logger.info(
            "guide participant connected with fresh state pid={}",
            participant_id,
        )

    async def close(self) -> None:
        self._closed.set()
        for participant_id in list(self._sessions):
            await self.release(participant_id)

    async def _evaluate(self, session: WorkflowSession) -> None:
        if not session.active or not session.connected:
            return
        async with session.lock:
            if not session.active or not session.connected or session.evaluation_active or session.user_turn_active:
                return
            step = self._workflow.step_by_id(session.step_id)
            if step.is_idle:
                return
            if session.ready_step_id == step.id:
                return
            session.evaluation_active = True
            last_frame_pts_us = session.last_frame_pts_us
            context_snapshot = dict(session.context)
            observation_log = list(session.observation_log)
            logger.debug(
                "guide eval begin pid={} step={} state={} mechanism={}",
                session.participant_id,
                step.id,
                session.step_state,
                step.mechanism,
            )
        try:
            try:
                iteration = await self._step_mechanisms.run(
                    participant_id=session.participant_id,
                    step=step,
                    context=context_snapshot,
                    observation_log=observation_log,
                    last_frame_pts_us=last_frame_pts_us,
                    event=StepEvent.periodic(step_state=session.step_state),
                )
            except FrameUnavailable:
                logger.debug("no fresh frame for participant {}", session.participant_id)
                return
            except Exception:
                logger.exception("step mechanism failed")
                return

            if iteration is None:
                return

            result = iteration.result
            ready_notice = ""
            reminder_notice = ""
            urgent_notice = ""
            async with session.lock:
                if not session.active or not session.connected or session.step_id != step.id:
                    return
                monitor_observations = [
                    observation for observation in iteration.observations if observation.kind == "step_monitor"
                ]
                for observation in monitor_observations:
                    if observation.frame_pts_us <= session.last_frame_pts_us:
                        return
                    session.last_frame_pts_us = observation.frame_pts_us
                    self._store_observation(
                        session,
                        step,
                        observation.text,
                        observation.frame_pts_us,
                    )
                self._merge_context(
                    session.context,
                    result.context_patch,
                    allowed=step.writable_fields,
                )
                if result.assistant_message:
                    session.pending_instruction = result.assistant_message
                    if result.speak:
                        urgent_notice = result.assistant_message
                logger.debug(
                    "guide eval result pid={} step={} state={} ready={} patch_keys={} assistant_message={!r} speak={}",
                    session.participant_id,
                    step.id,
                    result.step_state,
                    result.ready_to_advance,
                    sorted(result.context_patch),
                    result.assistant_message,
                    result.speak,
                )

                if self._workflow.advance_when_met(
                    step,
                    session.context,
                    ready_to_advance=result.ready_to_advance,
                ):
                    ready_notice = self._mark_ready_for_next(session, step)
                else:
                    session.step_state = result.step_state if result.step_state != "complete" else "started"
                    reminder_notice = self._reminder_due(session, step)

            if urgent_notice and not ready_notice:
                await self._maybe_notice(session, urgent_notice)
            if reminder_notice:
                await self._maybe_notice(session, reminder_notice, force=True)
            if ready_notice:
                await self._maybe_notice(session, ready_notice, force=True)
        finally:
            async with session.lock:
                session.evaluation_active = False

    async def _handle_voice_event(
        self,
        *,
        participant_id: str,
        transcript: str,
        session: WorkflowSession | None,
        step: WorkflowStep,
    ) -> str:
        if session is not None:
            async with session.lock:
                context = dict(session.context)
                observation_log = list(session.observation_log)
                step_state = session.step_state
                ready = session.ready_step_id == step.id
                workflow_active = session.active
        else:
            context = self._workflow.initial_context()
            observation_log = []
            step_state = "idle"
            ready = False
            workflow_active = False

        try:
            iteration = await self._step_mechanisms.run(
                participant_id=participant_id,
                step=step,
                context=context,
                observation_log=observation_log,
                last_frame_pts_us=session.last_frame_pts_us if session is not None else 0,
                event=StepEvent.voice(
                    transcript,
                    recent_turns=self._history.get(participant_id, []),
                    step_state=step_state,
                    ready=ready,
                    workflow_active=workflow_active,
                ),
            )
        except Exception:
            logger.exception("step mechanism voice event failed")
            return "I could not answer that just now. Please try again."

        if iteration is None:
            return "I could not answer that just now. Please try again."
        if iteration.result.context_patch:
            logger.warning(
                "discarding context patch from read-only voice event pid={} step={} keys={}",
                participant_id,
                step.id,
                sorted(iteration.result.context_patch),
            )

        if session is not None and workflow_active and iteration.observations:
            async with session.lock:
                if session.active == workflow_active and session.step_id == step.id:
                    for observation in iteration.observations:
                        self._store_observation(
                            session,
                            step,
                            observation.text,
                            observation.frame_pts_us,
                            kind=observation.kind,
                            question=observation.question,
                        )
        return iteration.result.assistant_message.strip() or "Done."

    def _merge_context(
        self,
        context: dict[str, Any],
        patch: dict[str, Any],
        *,
        allowed: set[str],
    ) -> None:
        for key, value in patch.items():
            if key not in allowed:
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            context[key] = value

    def _store_observation(
        self,
        session: WorkflowSession,
        step: WorkflowStep,
        text: str,
        frame_pts_us: int,
        *,
        kind: str = "step_monitor",
        question: str = "",
    ) -> None:
        if not text.strip():
            return
        session.observation_log.append(
            {
                "step_id": step.id,
                "step_name": step.name,
                "frame_pts_us": frame_pts_us,
                "observed_at_us": _now_us(),
                "kind": kind,
                "caption": text.strip(),
            }
        )
        if question:
            session.observation_log[-1]["question"] = question
        logger.debug(
            "vlm observation stored pid={} step={} frame_pts_us={} caption={!r}",
            session.participant_id,
            step.id,
            frame_pts_us,
            text[:240],
        )
        del session.observation_log[:-20]

    def _mark_ready_for_next(
        self,
        session: WorkflowSession,
        step: WorkflowStep,
    ) -> str:
        if session.ready_step_id == step.id:
            return ""
        session.ready_step_id = step.id
        session.step_state = "complete"
        session.pending_instruction = ""
        session.deferred_notice = ""
        session.reminder_count = self._workflow.max_reminders_per_step
        message = self._message(
            step.on_complete_message,
            session=session,
            step=step,
            fallback=f"Step {step.id}: {step.name} is ready.",
        )
        notice = message
        if self._workflow.next_step(step.id) is not None:
            notice = f"{message} Say next when you're ready."
        logger.info(
            "guide step ready pid={} step={} notice={!r}",
            session.participant_id,
            step.id,
            notice,
        )
        if self._workflow.next_step(step.id) is None:
            self._set_idle(session)
        return notice

    def _set_idle(self, session: WorkflowSession) -> None:
        previous_step = session.step_id
        self._history.pop(session.participant_id, None)
        session.active = False
        session.step_id = 0
        session.context = self._workflow.initial_context()
        session.last_frame_pts_us = 0
        session.ready_step_id = None
        session.step_state = "idle"
        session.step_started_us = 0
        session.last_reminder_us = 0
        session.reminder_count = 0
        session.pending_instruction = ""
        session.deferred_notice = ""
        session.observation_log.clear()
        logger.info(
            "guide idle pid={} previous_step={}",
            session.participant_id,
            previous_step,
        )

    def _reminder_due(self, session: WorkflowSession, step: WorkflowStep) -> str:
        if session.ready_step_id == step.id:
            return ""
        if self._workflow.condition_met(step.suppress_reminders_when, session.context):
            return ""
        if session.reminder_count >= self._workflow.max_reminders_per_step:
            return ""

        now_us = _now_us()
        interval_us = int(self._workflow.reminder_interval_s * 1_000_000)
        if now_us - session.last_reminder_us < interval_us:
            return ""

        session.reminder_count += 1
        session.last_reminder_us = now_us
        session.step_state = "needs_input"
        if session.pending_instruction.strip():
            logger.info(
                "guide reminder due pid={} step={} source=pending text={!r}",
                session.participant_id,
                step.id,
                session.pending_instruction,
            )
            return session.pending_instruction
        message = self._message(
            step.on_reminder_message or step.on_enter_message,
            session=session,
            step=step,
            fallback=f"Continue step {step.id}: {step.name}.",
        )
        logger.info(
            "guide reminder due pid={} step={} source=yaml text={!r}",
            session.participant_id,
            step.id,
            message,
        )
        return message

    async def _maybe_notice(
        self,
        session: WorkflowSession,
        text: str,
        *,
        force: bool = False,
    ) -> None:
        text = speech_text(text)
        if not text or not session.connected:
            return
        if session.user_turn_active:
            session.deferred_notice = text
            logger.debug(
                "guide notice deferred pid={} step={} text={!r}",
                session.participant_id,
                session.step_id,
                text,
            )
            return
        now_us = _now_us()
        min_interval_us = int(self._workflow.min_notice_interval_s * 1_000_000)
        if not force and now_us - session.last_notice_us < min_interval_us:
            logger.debug(
                "guide notice suppressed by interval pid={} step={} text={!r}",
                session.participant_id,
                session.step_id,
                text,
            )
            return
        session.last_notice_us = now_us
        session.last_notice_text = text
        logger.info(
            "guide notice pid={} step={} text={!r}",
            session.participant_id,
            session.step_id,
            text,
        )
        await self._notice(session.participant_id, text)

    def _begin_step(self, session: WorkflowSession, step: WorkflowStep) -> None:
        now_us = _now_us()
        session.step_id = step.id
        session.active = True
        session.last_frame_pts_us = 0
        session.ready_step_id = None
        session.step_state = "started"
        session.step_started_us = now_us
        session.last_reminder_us = now_us
        session.reminder_count = 0
        session.pending_instruction = ""
        session.deferred_notice = ""
        logger.info(
            "guide begin step pid={} step={} name={!r}",
            session.participant_id,
            step.id,
            step.name,
        )

    def _current_step(self, session: WorkflowSession | None) -> WorkflowStep:
        if session is not None:
            return self._workflow.step_by_id(session.step_id)
        return self._workflow.step_by_id(0)

    def _message(
        self,
        template: str,
        *,
        session: WorkflowSession,
        step: WorkflowStep,
        fallback: str,
    ) -> str:
        if not template.strip():
            return fallback
        return (
            speech_text(
                render_template(
                    template,
                    context=session.context,
                    step=step,
                    task=self._workflow.task,
                ).strip()
            )
            or fallback
        )

    def _record(self, participant_id: str, user: str, assistant: str) -> None:
        turns = self._history.setdefault(participant_id, [])
        turns.append((user, assistant))
        del turns[:-4]


def _advance_prefix(skipped: bool) -> str:
    return "Using reasonable defaults and moving on. " if skipped else ""


_COMMAND_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_QUESTION_STARTS = {
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
}
_DEFAULT_ADVANCE_EXAMPLES = (
    "next",
    "next step",
    "continue",
    "go on",
    "move on",
    "proceed",
    "proceed further",
    "let us proceed",
    "lets proceed",
    "let us proceed further",
    "lets proceed further",
    "i am ready",
    "im ready",
    "skip",
    "skip this",
    "skip this step",
)
_NEGATION_WORDS = {"not", "never", "dont", "no"}
_START_WORDS = {"begin", "guide", "help", "start", "walk"}
_STOP_WORDS = {"cancel", "end", "quit", "stop"}


def _idle_navigation_answer(
    text: str,
    session: WorkflowSession | None,
    workflow: WorkflowDefinition,
) -> str:
    if session is not None and session.active:
        return ""
    clean = _normalize_command(text)
    words = set(clean.split())
    if "next" not in words and not _looks_like_advance(text, workflow):
        return ""
    idle = workflow.step_by_id(0)
    start_hint = render_template(
        idle.on_enter_message,
        context=workflow.initial_context(),
        step=idle,
        task=workflow.task,
    ).strip()
    suffix = f" {start_hint}" if start_hint else ""
    return f"No guided workflow is active; the previous session is finished.{suffix}"


def _local_intent(
    text: str,
    workflow: WorkflowDefinition,
    *,
    active: bool,
) -> NavigationIntent | None:
    if _contains_negation(text):
        return None
    if active and _matches_any(text, workflow.stop_triggers):
        return NavigationIntent(intent="stop", confidence=1.0)
    if _matches_any(text, workflow.status_triggers):
        return NavigationIntent(intent="status", confidence=1.0)
    if not active and _matches_any(text, workflow.start_triggers):
        return NavigationIntent(intent="start", confidence=1.0)
    if active and _looks_like_advance(text, workflow):
        return NavigationIntent(
            intent="advance",
            skip_requested=_skip_requested(text),
            explicit_command=True,
            confidence=0.9,
        )
    return None


def _validated_model_intent(
    text: str,
    intent: NavigationIntent,
    workflow: WorkflowDefinition,
    *,
    active: bool,
) -> NavigationIntent:
    if intent.confidence < 0.75 or _contains_negation(text):
        return NavigationIntent()
    clean = _normalize_command(text)
    words = set(clean.split())
    if intent.intent == "advance" and active and intent.explicit_command and not _looks_like_question(text, clean):
        return intent
    if intent.intent == "start" and not active and words & _START_WORDS:
        return intent
    if intent.intent == "stop" and active and words & _STOP_WORDS:
        return intent
    if intent.intent == "status" and _looks_like_status_request(clean):
        return intent
    return NavigationIntent()


def _looks_like_advance(text: str, workflow: WorkflowDefinition) -> bool:
    clean = _normalize_command(text)
    if not clean or _looks_like_question(text, clean):
        return False
    examples = workflow.task.get("navigation_examples", {})
    configured = ()
    if isinstance(examples, dict):
        configured = tuple(str(item) for item in examples.get("advance", ()) or ())
    for phrase in (*configured, *_DEFAULT_ADVANCE_EXAMPLES):
        normalized = _normalize_command(phrase)
        if normalized and (clean == normalized or clean.endswith(f" {normalized}")):
            return True
    return False


def _looks_like_status_request(clean: str) -> bool:
    return "status" in clean.split() or "what step" in clean or "where are we" in clean or "workflow progress" in clean


def _contains_negation(text: str) -> bool:
    return bool(set(_normalize_command(text).split()) & _NEGATION_WORDS)


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    clean = _normalize_command(text)
    return any(
        normalized and (clean == normalized or normalized in clean)
        for normalized in (_normalize_command(phrase) for phrase in phrases)
    )


def _looks_like_question(original: str, clean: str) -> bool:
    first = clean.split(" ", 1)[0] if clean else ""
    return "?" in original or first in _QUESTION_STARTS


def _skip_requested(text: str) -> bool:
    words = _normalize_command(text).split()
    return "skip" in words or any(word.startswith("default") for word in words)


def _normalize_command(text: str) -> str:
    lowered = text.casefold().replace("'", "").replace("\N{RIGHT SINGLE QUOTATION MARK}", "")
    return " ".join(_COMMAND_CLEAN_RE.sub(" ", lowered).split())


def _now_us() -> int:
    return time.time_ns() // 1_000


__all__ = ["NoticeFn", "WorkflowGuide"]
