# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic participant-local state for guided physical workflows."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .spec import StateField, Step, Workflow

_TOKEN = re.compile(
    r"{{\s*([a-z][a-z0-9_-]*)\s*(?:\|\s*([a-z][a-z0-9_-]*))?\s*}}"
)


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """One state-machine event ready for runtime publication."""

    event: str
    step_id: str | None
    message: str
    state: dict[str, Any]


@dataclass(slots=True)
class WorkflowSession:
    """Mutable state belonging to exactly one connected participant."""

    participant_id: str
    state: dict[str, Any]
    active: bool = False
    step_id: str | None = None
    revision: int = 0
    next_tick: float = 0.0
    evidence_hits: int = 0
    notices: list[str] = field(default_factory=list)
    events: list[WorkflowEvent] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Deterministic result of one model-requested state commit."""

    accepted: bool
    complete: bool
    message: str
    revision: int


class WorkflowStore:
    """Own workflow sessions and enforce every state transition."""

    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._sessions: dict[str, WorkflowSession] = {}

    def get(self, participant_id: str) -> WorkflowSession:
        """Return or create one participant session."""

        if not participant_id.strip():
            raise ValueError("participant_id must not be empty")
        return self._sessions.setdefault(
            participant_id,
            WorkflowSession(
                participant_id=participant_id,
                state=self.workflow.initial_state(),
            ),
        )

    def find(self, participant_id: str) -> WorkflowSession | None:
        """Return an existing participant session without creating one."""

        return self._sessions.get(participant_id)

    def sessions(self) -> tuple[WorkflowSession, ...]:
        """Return all current participant sessions."""

        return tuple(self._sessions.values())

    def release(self, participant_id: str) -> WorkflowSession | None:
        """Discard all workflow state for a disconnected participant."""

        return self._sessions.pop(participant_id, None)

    def active_step(self, session: WorkflowSession) -> Step:
        """Return the current step or reject an idle workflow."""

        if not session.active or session.step_id is None:
            raise ValueError("workflow is idle")
        return self.workflow.step(session.step_id)

    def step_complete(self, session: WorkflowSession) -> bool:
        """Return whether the active step's predicate currently holds."""

        return self.active_step(session).is_complete(session.state)

    def start(self, session: WorkflowSession) -> str:
        """Start an idle workflow without erasing an active one."""

        if session.active:
            message = self.status(session)
            self._event(session, "workflow.start_noop", message)
            return message
        return self._restart(session, event="workflow.started")

    def restart(self, session: WorkflowSession) -> str:
        """Clear all progress and enter the first step."""

        return self._restart(session, event="workflow.restarted")

    def reset(self, session: WorkflowSession) -> str:
        """Clear progress and release foreground ownership."""

        session.state = self.workflow.initial_state()
        session.active = False
        session.step_id = None
        session.evidence_hits = 0
        session.notices.clear()
        session.revision += 1
        message = f"{self.workflow.name} guidance reset."
        self._event(session, "workflow.reset", message)
        return message

    def status(self, session: WorkflowSession) -> str:
        """Render a concise current-state response."""

        if not session.active or session.step_id is None:
            return f"{self.workflow.name} guidance is idle."
        step = self.workflow.step(session.step_id)
        suffix = (
            " Complete; say next when ready."
            if step.is_complete(session.state)
            else ""
        )
        return f"Current step: {step.title}.{suffix}"

    def advance(self, session: WorkflowSession, *, skip: bool) -> str:
        """Advance only after completion or an explicit skip."""

        step = self.active_step(session)
        complete = step.is_complete(session.state)
        if not complete and not skip:
            message = (
                f"{step.title} is not complete yet. "
                "Say skip if you want to move on anyway."
            )
            self._event(session, "workflow.advance_blocked", message)
            return message
        if skip:
            invalid = self._invalid_patch(step, step.state_on_skip)
            if invalid:
                raise ValueError(f"invalid skip state for {step.id}: {invalid}")
            session.state.update(copy.deepcopy(step.state_on_skip))
            session.revision += 1
        transition = self._transition(session, step)
        if not skip:
            self._event(session, "workflow.advanced", transition)
            return transition
        message = " ".join(
            item
            for item in (
                self._render(step.skip_message, session),
                transition,
            )
            if item
        )
        self._event(session, "workflow.skipped", message)
        return message

    def observe(
        self,
        session: WorkflowSession,
        observation: Any,
    ) -> None:
        """Update consecutive evidence without letting the model control it."""

        step = self.active_step(session)
        if step.evidence is None:
            return
        value = (
            observation
            if isinstance(observation, str)
            else json.dumps(observation, separators=(",", ":"))
        )
        matched = re.fullmatch(step.evidence.pattern, value.strip()) is not None
        session.evidence_hits = session.evidence_hits + 1 if matched else 0
        self._event(
            session,
            "step.evidence",
            (
                f"matched={str(matched).lower()} "
                f"consecutive={session.evidence_hits}/"
                f"{step.evidence.consecutive}"
            ),
        )

    def commit(
        self,
        session: WorkflowSession,
        updates: dict[str, Any],
        message: str,
    ) -> CommitResult:
        """Validate and atomically commit one active-step patch."""

        step = self.active_step(session)
        was_complete = step.is_complete(session.state)
        invalid = self._invalid_patch(step, updates)
        if invalid:
            self._event(session, "step.commit_rejected", invalid)
            return CommitResult(False, False, invalid, session.revision)
        if was_complete:
            self._event(session, "step.commit_noop", "state unchanged")
            return CommitResult(True, True, "state unchanged", session.revision)
        changes = {
            name: value
            for name, value in updates.items()
            if name not in session.state or session.state[name] != value
        }
        candidate = {**session.state, **changes}
        if (
            step.evidence is not None
            and step.is_complete(candidate)
            and session.evidence_hits < step.evidence.consecutive
        ):
            reason = (
                "completion evidence "
                f"{session.evidence_hits}/{step.evidence.consecutive}"
            )
            self._event(session, "step.commit_rejected", reason)
            return CommitResult(False, False, reason, session.revision)
        if not changes:
            self._event(session, "step.commit_noop", "state unchanged")
            return CommitResult(True, False, "state unchanged", session.revision)
        session.state.update(copy.deepcopy(changes))
        session.revision += 1
        complete = step.is_complete(session.state)
        notification = message.strip() if not complete else ""
        if notification:
            session.notices.append(notification)
        self._event(session, "step.commit", notification)
        if complete:
            notice = self._render(step.complete_message, session)
            if notice:
                session.notices.append(notice)
            self._event(session, "step.ready", notice)
        return CommitResult(True, complete, "state committed", session.revision)

    def drain_notices(self, session: WorkflowSession) -> tuple[str, ...]:
        """Drain pending spoken notices exactly once."""

        notices = tuple(session.notices)
        session.notices.clear()
        return notices

    def drain_events(self, session: WorkflowSession) -> tuple[WorkflowEvent, ...]:
        """Drain pending runtime records exactly once."""

        events = tuple(session.events)
        session.events.clear()
        return events

    def record(
        self,
        session: WorkflowSession,
        event: str,
        message: str = "",
    ) -> None:
        """Append a non-mutating workflow runtime record."""

        self._event(session, event, message)

    def _restart(self, session: WorkflowSession, *, event: str) -> str:
        session.state = self.workflow.initial_state()
        session.active = True
        session.notices.clear()
        session.evidence_hits = 0
        session.revision += 1
        message = self._enter(session, self.workflow.start_step)
        self._event(session, event, message)
        return message

    def _enter(self, session: WorkflowSession, step_id: str) -> str:
        session.step_id = step_id
        session.next_tick = 0.0
        session.evidence_hits = 0
        message = self._render(
            self.workflow.step(step_id).enter_message,
            session,
        )
        self._event(session, "step.enter", message)
        return message

    def _transition(self, session: WorkflowSession, step: Step) -> str:
        if step.next_step is None:
            session.active = False
            session.step_id = None
            message = self._render(self.workflow.complete_message, session)
            self._event(session, "workflow.complete", message)
            return message
        return self._enter(session, step.next_step)

    def _invalid_patch(self, step: Step, updates: dict[str, Any]) -> str:
        unknown = updates.keys() - set(step.writes)
        if unknown:
            return f"fields not writable in this step: {sorted(unknown)}"
        for name, value in updates.items():
            expected = self.workflow.state_fields[name]
            if not _valid_type(expected, value):
                return f"{name} must be {expected.type}"
        return ""

    def _event(
        self,
        session: WorkflowSession,
        event: str,
        message: str,
    ) -> None:
        session.events.append(
            WorkflowEvent(
                event=event,
                step_id=session.step_id,
                message=message,
                state=copy.deepcopy(session.state),
            )
        )

    @staticmethod
    def _render(text: str, session: WorkflowSession) -> str:
        def replace(match: re.Match[str]) -> str:
            name, formatter = match.groups()
            if name not in session.state:
                raise ValueError(f"message references missing state: {name}")
            value = session.state[name]
            if formatter is None:
                return str(value)
            if formatter == "temperature_c":
                return f"{_number(value)} degrees Celsius"
            if formatter == "duration":
                return _duration(int(value))
            raise ValueError(f"unknown message formatter: {formatter}")

        return _TOKEN.sub(replace, text).strip()


def _duration(seconds: int) -> str:
    minutes, remainder = divmod(seconds, 60)
    parts: list[str] = []
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if remainder or not parts:
        parts.append(
            f"{remainder} second" + ("s" if remainder != 1 else "")
        )
    return " and ".join(parts)


def _number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _valid_type(field: StateField, value: Any) -> bool:
    if field.type == "boolean":
        return isinstance(value, bool)
    if field.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field.type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    return isinstance(value, str)


def monotonic_now() -> float:
    """Return the scheduling clock through one patchable boundary."""

    return time.monotonic()


__all__ = [
    "CommitResult",
    "WorkflowEvent",
    "WorkflowSession",
    "WorkflowStore",
    "monotonic_now",
]
