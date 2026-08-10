# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground ownership and background membership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...runtime.events import emit
from .spec import ApplicationCatalog


@dataclass(slots=True)
class ApplicationState:
    foreground: list[str] = field(default_factory=list)
    background: set[str] = field(default_factory=set)
    revision: int = 0


class ApplicationSession(Protocol):
    participant_id: str
    applications: ApplicationState


class ApplicationOwnership:
    ROOT = "root"

    def __init__(self, spec: ApplicationCatalog) -> None:
        self.spec = spec

    def current(self, session: ApplicationSession) -> str:
        return session.applications.foreground[-1] if session.applications.foreground else self.ROOT

    def capture(self, session: ApplicationSession, app_id: str) -> None:
        self._require_mode(app_id, "foreground")
        if self.current(session) == app_id:
            return
        session.applications.foreground.append(app_id)
        session.applications.revision += 1
        emit(
            "application_manager.foreground.enter",
            participant_id=session.participant_id,
            application=app_id,
            depth=len(session.applications.foreground),
            revision=session.applications.revision,
        )

    def release(self, session: ApplicationSession, app_id: str) -> None:
        if self.current(session) != app_id:
            raise ValueError(f"{app_id} does not own the foreground")
        session.applications.foreground.pop()
        session.applications.revision += 1
        emit(
            "application_manager.foreground.exit",
            participant_id=session.participant_id,
            application=app_id,
            next_application=self.current(session),
            depth=len(session.applications.foreground),
            revision=session.applications.revision,
        )

    def start_background(self, session: ApplicationSession, app_id: str) -> bool:
        self._require_mode(app_id, "background")
        if app_id in session.applications.background:
            return False
        session.applications.background.add(app_id)
        session.applications.revision += 1
        emit(
            "application_manager.background.start",
            participant_id=session.participant_id,
            application=app_id,
            revision=session.applications.revision,
        )
        return True

    def stop_background(self, session: ApplicationSession, app_id: str) -> bool:
        if app_id not in session.applications.background:
            return False
        session.applications.background.remove(app_id)
        session.applications.revision += 1
        emit(
            "application_manager.background.stop",
            participant_id=session.participant_id,
            application=app_id,
            revision=session.applications.revision,
        )
        return True

    def is_background_active(self, session: ApplicationSession, app_id: str) -> bool:
        return app_id in session.applications.background

    def reset(self, session: ApplicationSession) -> None:
        foreground = tuple(session.applications.foreground)
        background = tuple(sorted(session.applications.background))
        session.applications.foreground.clear()
        session.applications.background.clear()
        session.applications.revision += 1
        emit(
            "application_manager.reset",
            participant_id=session.participant_id,
            foreground=foreground,
            background=background,
            revision=session.applications.revision,
        )

    def status(self, session: ApplicationSession) -> str:
        foreground = self.current(session)
        foreground_title = (
            "Root assistant" if foreground == self.ROOT else self.spec.application(foreground).title
        )
        background = sorted(session.applications.background)
        if not background:
            return f"{foreground_title} is in the foreground. No background applications are running."
        titles = ", ".join(self.spec.application(app_id).title for app_id in background)
        return f"{foreground_title} is in the foreground. Running in the background: {titles}."

    def _require_mode(self, app_id: str, mode: str) -> None:
        app = self.spec.application(app_id)
        if app.mode != mode:
            raise ValueError(f"{app_id} is {app.mode}, not {mode}")


__all__ = ["ApplicationOwnership", "ApplicationState"]
