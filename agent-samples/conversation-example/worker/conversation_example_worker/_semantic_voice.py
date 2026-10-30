# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early endpoint advice never vetoes the ordinary speaker-ASR endpoint."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from loguru import logger
from xr_ai_voice import TranscriptUpdate, VoiceOutputState, VoiceTurnAction

from ._semantic_turns import _QUESTIONS, _is_clear_filler, _Policy

_FINAL_CAPACITY = 64


@dataclass
class _Source:
    epoch: str
    segment: int = -1
    revision: int = -1
    text: str = ""
    prefix: str = ""
    timestamp_us: int = 0
    final: bool = False
    attempted: tuple[int, int] | None = None
    admission_sources: OrderedDict[str, str] = field(default_factory=OrderedDict)


class _ClefVoicePolicy:
    def __init__(self, decisions, policy: _Policy | None = None):
        self._policy = policy or _Policy()
        self.settle_s = self._policy.stable_s
        self._decisions = decisions
        self._sources: dict[str, _Source] = {}
        self._next_request = 0
        self._pending_finals: dict[str, deque[VoiceTurnAction]] = {}

    def reset(self, participant_id: str) -> None:
        self._sources.pop(participant_id, None)
        self._pending_finals.pop(participant_id, None)

    def update(self, update: TranscriptUpdate) -> None:
        source = self._sources.get(update.participant_id)
        if source is None or source.epoch != update.source_epoch:
            if source is None and len(self._sources) >= self._policy.max_sessions:
                raise ValueError("early endpoint source capacity exceeded")
            self.reset(update.participant_id)
            source = self._sources[update.participant_id] = _Source(update.source_epoch)
        if update.segment_id < source.segment:
            return
        if update.segment_id != source.segment:
            source.segment = update.segment_id
            source.revision = -1
            source.prefix = ""
        if update.revision <= source.revision:
            return
        source.revision = update.revision
        source.text = update.text.strip()
        source.timestamp_us = update.timestamp_us
        source.final = update.final
        if update.final:
            # Retain native endpoints independently of later partial snapshots.
            # This synchronous path never asks the optional model for permission.
            pending = self._pending_finals.setdefault(update.participant_id, deque())
            text = self._tail(update.participant_id, source)
            if text is not None:
                if len(pending) >= _FINAL_CAPACITY:
                    raise ValueError("ordinary endpoint delivery overloaded")
                pending.extend(self._submit(update.participant_id, source, text, reason="vad_admit"))

    def output(self, state: VoiceOutputState) -> None:
        # Routing, interruption and response ownership retain the normal sample behavior.
        pass

    def _submit(self, pid: str, source: _Source, text: str, *, reason: str) -> list[VoiceTurnAction]:
        source.prefix = source.text
        self._next_request += 1
        request_id = f"semantic:{source.epoch}:{self._next_request}"
        source.admission_sources[request_id] = "Clef" if reason == "early_admit" else "VAD"
        # Retain queued finals plus recently dispatched diagnostics; cancelled
        # queries may never reach the runtime subscriber that consumes these.
        while len(source.admission_sources) > 2 * _FINAL_CAPACITY:
            source.admission_sources.popitem(last=False)
        logger.info("voice endpoint pid={!r} action={} request_id={!r}", pid, reason, request_id)
        return [VoiceTurnAction("submit", request_id, text, source.timestamp_us)]

    def take_admission_source(self, participant_id: str, request_id: str | None) -> str | None:
        """Consume diagnostic provenance for an actually delivered sample query."""
        source = self._sources.get(participant_id)
        return source.admission_sources.pop(request_id, None) if source is not None else None

    def _tail(self, participant_id: str, source: _Source) -> str | None:
        if source.prefix and not source.text.startswith(source.prefix):
            logger.warning("ASR revised already-admitted speech pid={!r}; not replaying it", participant_id)
            return None
        text = source.text[len(source.prefix) :].strip()
        if not text.strip(" \t\n.,!?;:…"):
            return None
        if _is_clear_filler(text):
            logger.info("voice endpoint pid={!r} action=filler_drop final={}", participant_id, source.final)
            return None
        return text

    async def decide(self, participant_id: str) -> list[VoiceTurnAction]:
        pending = self._pending_finals.pop(participant_id, None)
        if pending is not None:
            return list(pending)
        source = self._sources.get(participant_id)
        if source is None or source.final:
            return []
        version = (source.segment, source.revision)
        if source.attempted == version:
            return []
        source.attempted = version
        text = self._tail(participant_id, source)
        if text is None:
            return []
        if len(text) > self._policy.max_chars:
            logger.debug("early endpoint skipped oversized input pid={!r}; awaiting ordinary endpoint", participant_id)
            return []
        try:
            async with asyncio.timeout(self._policy.decision_timeout_s):
                result = await self._decisions.decide(
                    {"transcript": text},
                    _QUESTIONS,
                    timeout=self._policy.decision_timeout_s,
                )
            answer = result.answers["ready"]
            if (
                set(result.answers) != {"ready"}
                or answer.choice not in {"yes", "no"}
                or isinstance(answer.confidence, bool)
                or not math.isfinite(answer.confidence)
                or not 0 <= answer.confidence <= 1
            ):
                raise ValueError("malformed early endpoint answer")
        except asyncio.CancelledError:
            source.attempted = None
            raise
        except Exception:  # noqa: BLE001 - optional model advice must never veto normal endpointing
            logger.warning("early endpoint unavailable pid={!r}; ordinary endpoint remains enabled", participant_id)
            return []
        if self._sources.get(participant_id) is not source or (source.segment, source.revision) != version:
            return []
        logger.debug(
            "early endpoint pid={!r} choice={} score={:.4f} threshold={:.4f}",
            participant_id,
            answer.choice,
            answer.confidence,
            self._policy.min_confidence,
        )
        if answer.choice == "yes" and answer.confidence >= self._policy.min_confidence:
            return self._submit(participant_id, source, text, reason="early_admit")
        return []
