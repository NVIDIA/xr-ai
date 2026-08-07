# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground stack and background application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..runtime.events import emit
from .spec import DesktopSpec


@dataclass(slots=True)
class DesktopState:
    foreground: list[str] = field(default_factory=list)
    background: set[str] = field(default_factory=set)
    revision: int = 0


class DesktopSession(Protocol):
    participant_id: str
    desktop: DesktopState


class DesktopRuntime:
    ROOT = "root"

    def __init__(self, spec: DesktopSpec) -> None:
        self.spec = spec

    def current(self, session: DesktopSession) -> str:
        return session.desktop.foreground[-1] if session.desktop.foreground else self.ROOT

    def capture(self, session: DesktopSession, app_id: str) -> None:
        self._require_mode(app_id, "foreground")
        if self.current(session) == app_id:
            return
        session.desktop.foreground.append(app_id)
        session.desktop.revision += 1
        emit(
            "desktop.foreground.enter",
            participant_id=session.participant_id,
            application=app_id,
            depth=len(session.desktop.foreground),
            revision=session.desktop.revision,
        )

    def release(self, session: DesktopSession, app_id: str) -> None:
        if self.current(session) != app_id:
            raise ValueError(f"{app_id} does not own the foreground")
        session.desktop.foreground.pop()
        session.desktop.revision += 1
        emit(
            "desktop.foreground.exit",
            participant_id=session.participant_id,
            application=app_id,
            next_application=self.current(session),
            depth=len(session.desktop.foreground),
            revision=session.desktop.revision,
        )

    def start_background(self, session: DesktopSession, app_id: str) -> bool:
        self._require_mode(app_id, "background")
        if app_id in session.desktop.background:
            return False
        session.desktop.background.add(app_id)
        session.desktop.revision += 1
        emit(
            "desktop.background.start",
            participant_id=session.participant_id,
            application=app_id,
            revision=session.desktop.revision,
        )
        return True

    def stop_background(self, session: DesktopSession, app_id: str) -> bool:
        if app_id not in session.desktop.background:
            return False
        session.desktop.background.remove(app_id)
        session.desktop.revision += 1
        emit(
            "desktop.background.stop",
            participant_id=session.participant_id,
            application=app_id,
            revision=session.desktop.revision,
        )
        return True

    def is_background_active(self, session: DesktopSession, app_id: str) -> bool:
        return app_id in session.desktop.background

    def reset(self, session: DesktopSession) -> None:
        foreground = tuple(session.desktop.foreground)
        background = tuple(sorted(session.desktop.background))
        session.desktop.foreground.clear()
        session.desktop.background.clear()
        session.desktop.revision += 1
        emit(
            "desktop.reset",
            participant_id=session.participant_id,
            foreground=foreground,
            background=background,
            revision=session.desktop.revision,
        )

    def status(self, session: DesktopSession) -> str:
        foreground = self.current(session)
        foreground_title = (
            "Root assistant" if foreground == self.ROOT else self.spec.application(foreground).title
        )
        background = sorted(session.desktop.background)
        if not background:
            return f"{foreground_title} is in the foreground. No background applications are running."
        titles = ", ".join(self.spec.application(app_id).title for app_id in background)
        return f"{foreground_title} is in the foreground. Running in the background: {titles}."

    def _require_mode(self, app_id: str, mode: str) -> None:
        app = self.spec.application(app_id)
        if app.mode != mode:
            raise ValueError(f"{app_id} is {app.mode}, not {mode}")


__all__ = ["DesktopRuntime", "DesktopState"]
