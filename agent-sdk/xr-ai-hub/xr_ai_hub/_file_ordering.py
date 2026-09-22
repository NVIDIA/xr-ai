# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-session ordering for the file IPC lane."""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from ._types import FileMessage

# Lifecycle and file IPC use separate sockets. Five seconds is ample ordering
# grace while staying well below the default 60-second transfer timeout.
_PENDING_FILE_TTL_S = 5.0
_DEPARTED_SESSION_TTL_S = 60.0
_MAX_DEPARTED_SESSIONS = 1024


class FileRoute(Enum):
    """Disposition of one file received on the file IPC lane."""

    READY = auto()
    BUFFERED = auto()
    INACTIVE = auto()
    FULL = auto()


@dataclass(slots=True)
class _PendingFile:
    message: FileMessage
    deadline_s: float


class FileSessionOrderer:
    """Correlate files with lifecycle events arriving on another socket."""

    def __init__(
        self,
        max_pending: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        pending_ttl_s: float = _PENDING_FILE_TTL_S,
        departure_ttl_s: float = _DEPARTED_SESSION_TTL_S,
        max_departures: int = _MAX_DEPARTED_SESSIONS,
    ) -> None:
        self._max_pending = max_pending
        self._clock = clock
        self._pending_ttl_s = pending_ttl_s
        self._departure_ttl_s = departure_ttl_s
        self._max_departures = max_departures
        self._active: dict[str, str] = {}
        self._pending: deque[_PendingFile] = deque()
        self._departed: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._expired: deque[FileMessage] = deque()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def route(self, message: FileMessage) -> FileRoute:
        now_s = self._clock()
        self._prune(now_s)
        session = (message.participant_id, message.participant_session_id)
        if message.participant_session_id and session in self._departed:
            return FileRoute.INACTIVE
        active_session = self._active.get(message.participant_id)
        if active_session is not None and (
            not message.participant_session_id
            or message.participant_session_id == active_session
        ):
            return FileRoute.READY
        if len(self._pending) >= self._max_pending:
            return FileRoute.FULL
        self._pending.append(
            _PendingFile(message, now_s + self._pending_ttl_s),
        )
        return FileRoute.BUFFERED

    def participant_joined(
        self,
        participant_id: str,
        participant_session_id: str,
    ) -> tuple[list[FileMessage], list[FileMessage]]:
        now_s = self._clock()
        self._prune(now_s)
        previous_session = self._active.get(participant_id)
        if previous_session and previous_session != participant_session_id:
            self._record_departure(participant_id, previous_session, now_s)
        self._active[participant_id] = participant_session_id
        self._departed.pop((participant_id, participant_session_id), None)
        ready: list[FileMessage] = []
        inactive: list[FileMessage] = []
        retained: deque[_PendingFile] = deque()
        while self._pending:
            pending = self._pending.popleft()
            message = pending.message
            if message.participant_id != participant_id:
                retained.append(pending)
            elif (
                not message.participant_session_id
                or message.participant_session_id == participant_session_id
            ):
                ready.append(message)
            elif (
                participant_id,
                message.participant_session_id,
            ) in self._departed:
                inactive.append(message)
            else:
                retained.append(pending)
        self._pending = retained
        return ready, inactive

    def participant_left(
        self,
        participant_id: str,
        participant_session_id: str,
    ) -> list[FileMessage]:
        now_s = self._clock()
        self._prune(now_s)
        if self._active.get(participant_id) == participant_session_id:
            self._active.pop(participant_id, None)
        if participant_session_id:
            self._record_departure(participant_id, participant_session_id, now_s)
        discarded: list[FileMessage] = []
        retained: deque[_PendingFile] = deque()
        while self._pending:
            pending = self._pending.popleft()
            message = pending.message
            if (
                message.participant_id == participant_id
                and message.participant_session_id == participant_session_id
            ):
                discarded.append(message)
            else:
                retained.append(pending)
        self._pending = retained
        return discarded

    def expire(self) -> list[FileMessage]:
        self._prune(self._clock())
        expired = list(self._expired)
        self._expired.clear()
        return expired

    def seconds_until_expiry(self) -> float | None:
        if self._expired:
            return 0.0
        if not self._pending:
            return None
        return max(0.0, self._pending[0].deadline_s - self._clock())

    def _record_departure(
        self,
        participant_id: str,
        participant_session_id: str,
        now_s: float,
    ) -> None:
        session = (participant_id, participant_session_id)
        self._departed[session] = now_s + self._departure_ttl_s
        self._departed.move_to_end(session)
        while len(self._departed) > self._max_departures:
            self._departed.popitem(last=False)

    def _prune(self, now_s: float) -> None:
        while self._pending and self._pending[0].deadline_s <= now_s:
            self._expired.append(self._pending.popleft().message)
        while self._departed:
            session, deadline_s = next(iter(self._departed.items()))
            if deadline_s > now_s:
                break
            self._departed.pop(session)
