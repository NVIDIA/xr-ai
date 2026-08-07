# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fan raw speech transcripts and ticks to active background applications."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from ..runtime.state import Session

Notice = Callable[[str, str], Awaitable[None]]
TextOutput = Callable[[str, str, str], Awaitable[None]]


class BackgroundApplication(Protocol):
    @property
    def app_id(self) -> str: ...

    async def on_transcription(self, session: Session, text: str, trace_id: str) -> None: ...

    async def tick(self, session: Session) -> None: ...

    async def release(self, session: Session) -> None: ...


class BackgroundRegistry:
    def __init__(self) -> None:
        self._applications: dict[str, BackgroundApplication] = {}

    def register(self, application: BackgroundApplication) -> None:
        if application.app_id in self._applications:
            raise ValueError(f"background application already registered: {application.app_id}")
        self._applications[application.app_id] = application

    async def on_transcription(self, session: Session, text: str, trace_id: str) -> None:
        for app_id in tuple(session.desktop.background):
            await self._applications[app_id].on_transcription(session, text, trace_id)

    async def tick(self, session: Session) -> None:
        for app_id in tuple(session.desktop.background):
            await self._applications[app_id].tick(session)

    async def release(self, session: Session) -> None:
        for application in self._applications.values():
            await application.release(session)


__all__ = ["BackgroundApplication", "BackgroundRegistry", "Notice", "TextOutput"]
