# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned Clef policy adapter for the opt-in public voice boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger
from xr_ai_voice import TranscriptUpdate, VoiceOutputState, VoiceTurnAction

from ._semantic_turns import _Policy, _TurnController

_REQUEST_PREFIX = "semantic:"


@dataclass
class _Source:
    epoch: str
    segment: int = -1
    revision: int = -1
    logical: int = -1
    prefix: str = ""
    snapshot: str = ""
    tail_open: bool = False
    pending_pts: int | None = None
    latest_pts: int = 0
    request_pts: dict[str, int] = field(default_factory=dict)
    request_text: dict[str, str] = field(default_factory=dict)


class _ClefVoicePolicy:
    def __init__(self, decisions, policy: _Policy | None = None, *, allow_unspoken_merge: bool = False):
        policy = policy or _Policy()
        self.settle_s = policy.stable_s
        self._controller = _TurnController(decisions, policy, request_prefix=_REQUEST_PREFIX)
        self._sources: dict[str, _Source] = {}
        self._pending: dict[str, list] = {}
        self._allow_unspoken_merge = allow_unspoken_merge

    def reset(self, participant_id: str) -> None:
        source = self._sources.pop(participant_id, None)
        if source is not None:
            self._controller.reset(participant_id, source.epoch)
        self._pending.pop(participant_id, None)

    def update(self, update: TranscriptUpdate) -> None:
        pid = update.participant_id
        source = self._sources.get(pid)
        if source is None or source.epoch != update.source_epoch:
            self.reset(pid)
            self._controller.open(pid, update.source_epoch)
            source = self._sources[pid] = _Source(update.source_epoch)
        if update.segment_id < source.segment:
            return
        if update.segment_id != source.segment:
            source.segment = update.segment_id
            source.revision = -1
            source.prefix = ""
            source.tail_open = False
        if update.revision <= source.revision:
            return
        source.revision = update.revision
        text = update.text.strip()
        if source.prefix and not text.startswith(source.prefix):
            # A correction to already-admitted text cannot silently become a
            # second execution. Keep it visible for acoustic qualification.
            logger.warning("post-admission ASR correction needs clarification pid={!r} text={!r}", pid, text)
            tail = ""
        else:
            tail = text[len(source.prefix) :].strip()
        source.snapshot = text
        source.latest_pts = update.timestamp_us
        if not tail.strip(" \t\n.,!?;:…"):
            if not source.tail_open:
                return
            # Retraction must replace pending text and invalidate an awaiting
            # decision, even when the new snapshot has no admissible words.
            tail = ""
        if not source.tail_open:
            source.logical += 1
            source.tail_open = True
        if source.pending_pts is None:
            source.pending_pts = update.timestamp_us
        self._controller.feed(
            pid,
            source.epoch,
            segment=source.logical,
            revision=update.revision,
            text=tail,
            now=asyncio.get_running_loop().time(),
        )

    def output(self, state: VoiceOutputState) -> None:
        source = self._sources.get(state.participant_id)
        if source is None:
            return
        actions = self._controller.response(
            state.participant_id,
            source.epoch,
            state.request_id,
            "playing" if state.stage == "committed" else "done",
            now=asyncio.get_running_loop().time(),
        )
        if actions:
            self._pending.setdefault(state.participant_id, []).extend(actions)

    def effects_started(self, participant_id: str, request_id: str) -> None:
        """Authorize this request before effects; reject replaced speech work."""
        # Opt-in typed input uses SDK-generated UUIDs, outside this namespace.
        # Keep speech identity recognizable even after its source has reset.
        if not request_id.startswith(_REQUEST_PREFIX):
            return
        source = self._sources.get(participant_id)
        if source is None or not self._controller.mark_effects(participant_id, source.epoch, request_id):
            raise asyncio.CancelledError("semantic request was replaced or reset before effects")

    def participant_text(self, participant_id: str, request_id: str, fallback: str) -> str:
        """Return admitted participant words, excluding synthetic routing context."""
        source = self._sources.get(participant_id)
        if not request_id.startswith(_REQUEST_PREFIX):
            return fallback
        if source is None or request_id not in source.request_text:
            raise asyncio.CancelledError("semantic participant text is no longer available")
        return source.request_text[request_id]

    async def decide(self, participant_id: str) -> list[VoiceTurnAction]:
        source = self._sources.get(participant_id)
        if source is None:
            return []
        classified = await self._controller.poll(
            participant_id,
            source.epoch,
            now=asyncio.get_running_loop().time(),
        )
        if self._sources.get(participant_id) is not source:
            return []
        if classified and any(a.kind != "hold" for a in classified):
            source.prefix = source.snapshot
            source.tail_open = False
            for a in classified:
                if a.kind in {"dispatch", "queue"}:
                    source.request_pts[a.request_id] = (
                        source.pending_pts if source.pending_pts is not None else source.latest_pts
                    )
                    source.request_text[a.request_id] = a.text

            source.pending_pts = None
        live = self._controller.live_request_ids(participant_id, source.epoch)
        for retired in source.request_pts.keys() - live:
            source.request_pts.pop(retired)
            source.request_text.pop(retired, None)
        active = self._controller.active_request_id(participant_id, source.epoch)
        actions = [
            action
            for action in self._pending.pop(participant_id, []) + classified
            if action.kind != "dispatch" or action.request_id == active
        ]
        result = []
        for action in actions:
            if action.kind in {"supersede", "interrupt"}:
                result.append(VoiceTurnAction("interrupt", action.request_id))
            elif action.kind == "dispatch":
                text = action.text
                if action.previous_text:
                    text = (
                        "User follow-up (new intent): "
                        + text
                        + "\nEarlier request, context only; do not repeat its actions: "
                        + action.previous_text
                    )
                result.append(
                    VoiceTurnAction(
                        "submit",
                        action.request_id,
                        text,
                        source.request_pts.get(action.request_id, source.latest_pts),
                    )
                )
                if not self._allow_unspoken_merge:
                    self.effects_started(participant_id, action.request_id)
            elif action.kind == "hold":
                logger.debug("semantic voice held pid={!r} reason={}", participant_id, action.reason)
        return result
