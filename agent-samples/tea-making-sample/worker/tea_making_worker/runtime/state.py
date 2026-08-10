# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-local state machine; policy remains entirely in workflow YAML."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..applications.manager.runtime import ApplicationState
from ..spec import StateField, Step, Workflow
from .events import emit
from .render import render_message


@dataclass(slots=True)
class Session:
    participant_id: str
    state: dict[str, Any]
    applications: ApplicationState = field(default_factory=ApplicationState)
    active: bool = False
    step_id: str | None = None
    revision: int = 0
    next_tick: float = 0.0
    notices: list[str] = field(default_factory=list)
    evidence_hits: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(frozen=True, slots=True)
class CommitResult:
    accepted: bool
    complete: bool
    message: str
    revision: int


class SessionStore:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._sessions: dict[str, Session] = {}

    def get(self, participant_id: str) -> Session:
        return self._sessions.setdefault(
            participant_id,
            Session(participant_id=participant_id, state=self.workflow.initial_state()),
        )

    def active(self) -> tuple[Session, ...]:
        return tuple(session for session in self._sessions.values() if session.active)

    def sessions(self) -> tuple[Session, ...]:
        return tuple(self._sessions.values())

    def step_complete(self, session: Session) -> bool:
        return self._active_step(session).is_complete(session.state)

    def release(self, participant_id: str) -> None:
        self._sessions.pop(participant_id, None)

    def start(self, session: Session) -> str:
        if session.active:
            emit(
                "workflow.start_noop",
                participant_id=session.participant_id,
                step=session.step_id,
                revision=session.revision,
            )
            return self.status(session)
        return self._restart(session)

    def restart(self, session: Session) -> str:
        return self._restart(session)

    def _restart(self, session: Session) -> str:
        session.state = self.workflow.initial_state()
        session.active = True
        session.notices.clear()
        session.evidence_hits = 0
        session.revision += 1
        return self._enter(session, self.workflow.start_step)

    def reset(self, session: Session) -> str:
        session.state = self.workflow.initial_state()
        session.active = False
        session.step_id = None
        session.notices.clear()
        session.evidence_hits = 0
        session.revision += 1
        emit("workflow.reset", participant_id=session.participant_id, revision=session.revision)
        return f"{self.workflow.name} guidance reset."

    def observe(self, session: Session, observation: Any, trace_id: str) -> None:
        step = self._active_step(session)
        if step.evidence is None:
            return
        value = observation if isinstance(observation, str) else json.dumps(observation, separators=(",", ":"))
        matched = re.fullmatch(step.evidence.pattern, value.strip()) is not None
        session.evidence_hits = session.evidence_hits + 1 if matched else 0
        emit(
            "step.evidence",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            matched=matched,
            consecutive=session.evidence_hits,
            required=step.evidence.consecutive,
            observation=value,
        )

    def status(self, session: Session) -> str:
        if not session.active or session.step_id is None:
            return f"{self.workflow.name} guidance is idle."
        step = self.workflow.step(session.step_id)
        suffix = " Complete; say next when ready." if step.is_complete(session.state) else ""
        return f"Current step: {step.title}.{suffix}"

    def advance(self, session: Session, *, skip: bool) -> str:
        step = self._active_step(session)
        complete = step.is_complete(session.state)
        if not complete and not skip:
            return f"{step.title} is not complete yet. Say skip if you want to move on anyway."
        if skip:
            invalid = self._invalid_patch(step, step.state_on_skip)
            if invalid:
                raise ValueError(f"invalid skip state for {step.id}: {invalid}")
            session.state.update(copy.deepcopy(step.state_on_skip))
        transition = self._transition(session, step)
        if not skip:
            return transition
        return " ".join(item for item in (self._render(step.skip_message, session), transition) if item)

    def commit(self, session: Session, updates: dict[str, Any], message: str) -> CommitResult:
        step = self._active_step(session)
        was_complete = step.is_complete(session.state)
        invalid = self._invalid_patch(step, updates)
        if invalid:
            emit(
                "step.commit_rejected",
                participant_id=session.participant_id,
                step=step.id,
                reason=invalid,
            )
            return CommitResult(False, False, invalid, session.revision)
        if was_complete:
            emit(
                "step.commit_noop",
                participant_id=session.participant_id,
                step=step.id,
                revision=session.revision,
                complete=True,
                attempted_updates=updates,
                attempted_notification=message.strip(),
            )
            return CommitResult(True, True, "state unchanged", session.revision)
        changes = {
            name: value for name, value in updates.items() if name not in session.state or session.state[name] != value
        }
        candidate = {**session.state, **changes}
        if (
            step.evidence is not None
            and step.is_complete(candidate)
            and session.evidence_hits < step.evidence.consecutive
        ):
            reason = f"completion evidence {session.evidence_hits}/{step.evidence.consecutive}"
            emit(
                "step.commit_rejected",
                participant_id=session.participant_id,
                step=step.id,
                reason=reason,
            )
            return CommitResult(False, False, reason, session.revision)
        complete = False
        if not changes:
            emit(
                "step.commit_noop",
                participant_id=session.participant_id,
                step=step.id,
                revision=session.revision,
                complete=False,
                attempted_notification=message.strip(),
            )
            return CommitResult(True, complete, "state unchanged", session.revision)
        session.state.update(copy.deepcopy(changes))
        session.revision += 1
        complete = step.is_complete(session.state)
        notification = message.strip() if not complete else ""
        if notification:
            session.notices.append(notification)
        emit(
            "step.commit",
            participant_id=session.participant_id,
            step=step.id,
            revision=session.revision,
            updates=changes,
            complete=complete,
            notification=notification,
            ignored_notification=message.strip() if complete else "",
        )
        if complete and not was_complete:
            notice = self._render(step.complete_message, session)
            if notice:
                session.notices.append(notice)
            emit(
                "step.ready",
                participant_id=session.participant_id,
                step=step.id,
                revision=session.revision,
                notification=notice,
            )
        return CommitResult(True, complete, "state committed", session.revision)

    def drain_notices(self, session: Session) -> tuple[str, ...]:
        notices = tuple(session.notices)
        session.notices.clear()
        return notices

    def _active_step(self, session: Session) -> Step:
        if not session.active or session.step_id is None:
            raise ValueError("workflow is idle")
        return self.workflow.step(session.step_id)

    def _enter(self, session: Session, step_id: str) -> str:
        session.step_id = step_id
        session.next_tick = 0.0
        session.evidence_hits = 0
        step = self.workflow.step(step_id)
        emit(
            "step.enter",
            participant_id=session.participant_id,
            step=step.id,
            revision=session.revision,
        )
        return self._render(step.enter_message, session)

    def _transition(self, session: Session, step: Step) -> str:
        if step.next_step is None:
            session.active = False
            session.step_id = None
            emit("workflow.complete", participant_id=session.participant_id, revision=session.revision)
            return self._render(self.workflow.complete_message, session)
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

    @staticmethod
    def _render(text: str, session: Session) -> str:
        return render_message(text, session.state)


def _valid_type(field: StateField, value: Any) -> bool:
    if field.type == "boolean":
        return isinstance(value, bool)
    if field.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


__all__ = ["CommitResult", "Session", "SessionStore"]
