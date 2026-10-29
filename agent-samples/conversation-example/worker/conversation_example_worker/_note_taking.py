# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own participant-authored notes as a focused route destination."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from loguru import logger
from xr_ai_conversation import ConversationContextUpdate
from xr_ai_interaction import ResponseOutput, ResponseOutputs
from xr_ai_models import DecisionQuestion, DecisionService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutableAgent, RoutingAid
from xr_ai_runtime import AgentRuntime, RuntimeClosedError, Topic

from ._decisions import decide
from ._tasks import wait_despite_cancellation

ContextPublisher = Callable[[str, ConversationContextUpdate], Awaitable[None]]
Emit = Callable[[str], Awaitable[None]]

_CONTROL = (Path(__file__).parent / "prompts/note_control.txt").read_text(encoding="utf-8").strip()
# Static, so the routing request stays the same size however many notes are saved.
_ROUTING = (
    "A participant-scoped note-taking destination. It starts only when explicitly requested, captures "
    "user-authored notes, and stays focused until the participant finishes, including across a question "
    "answered by another destination. It does not provide general knowledge or inspect the current camera."
)
_FOCUSED_ACTIONS = {
    "capture": (
        "A statement supplied as content during focused notes, or explicitly dictated words to save. "
        "The participant is not asking for an answer or asking the assistant to perform the stated task."
    ),
    "readback": "The participant wants to review notes already captured.",
    "finish": "The participant clearly wants to end note-taking and consume this request.",
    "handoff": (
        "The participant asks the assistant for information, an explanation, advice, or an action, "
        "rather than asking to record those words. A question seeking an answer is handoff even while focused."
    ),
}
_UNFOCUSED_ACTIONS = {
    **_FOCUSED_ACTIONS,
    "start": "The participant explicitly wants to begin a note-taking session.",
    "other": "The current request is not a note action or an explicit request to save content.",
}
_NOTED = "Noted."
_NOT_SHARED = "I saved that note here, but couldn't share it with general conversation."
_STARTED = "I’m ready to take notes."
_FINISHED = "Note-taking is finished."
_NO_NOTES = "There are no notes in this session yet."
_NO_ANSWER = DestinationResult("answer", "")


class NoteTakingAgent(RoutableAgent):
    """Keep participant-authored notes in an explicit, focused session.

    Every note is kept until the participant leaves and is read back in full. Each saved
    note is also published as one incremental context update, so general conversation
    sees recent notes verbatim and older ones in its rolling summary.
    """

    def __init__(
        self,
        decisions: DecisionService,
        *,
        runtime: AgentRuntime,
        output_topic: Topic[ResponseOutput],
        publish_note: ContextPublisher | None = None,
        effects_started: Callable[[str, str], None] | None = None,
        participant_text: Callable[[str, str, str], str] | None = None,
    ) -> None:
        self._decisions = decisions
        self._publish_note = publish_note
        self._effects_started = effects_started
        self._participant_text = participant_text
        self._outputs = ResponseOutputs(runtime, output_topic, source="notes", closed_errors=(RuntimeClosedError,))
        self._pending: dict[str, tuple[str, str, bool]] = {}
        self._notes: dict[str, list[str]] = {}

    # RoutableAgent contract.

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid(name="notes", description=_ROUTING)

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        action = await self._classify(turn, focused=focused)
        # Saved notes stay readable after the session ends; readback does not restart it.
        if action == "handoff" or (not focused and action not in {"start", "capture", "readback"}):
            return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, action, focused)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        if self._effects_started is not None:
            self._effects_started(turn.participant_id, turn.turn_id)
        pending = self._pending.pop(turn.participant_id, None)
        if pending is None or pending[0] != turn.turn_id:
            return _NO_ANSWER
        action, was_focused = pending[1:]

        async def emit(text: str) -> None:
            async with self._outputs.reply(
                turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us, started=started
            ) as reply:
                await reply.send(text)

        if action == "capture":
            note = (
                self._participant_text(turn.participant_id, turn.turn_id, turn.query)
                if self._participant_text is not None else turn.query
            )
            text = await self._save_note(turn.participant_id, note)
            await emit(text)
            return DestinationResult("focus", text)
        if action == "start":
            await emit(_STARTED)
            return DestinationResult("focus", _STARTED)
        if action == "finish":
            await emit(_FINISHED)
            return DestinationResult("release", _FINISHED)
        if action == "readback":
            text = self._read_back(turn.participant_id)
            await emit(text)
            return DestinationResult("focus" if was_focused else "answer", text)
        raise ValueError(f"invalid note action {action!r}")

    async def release(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)

    async def participant_left(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)
        self._notes.pop(participant_id, None)

    # Note actions.

    async def _classify(self, turn: AgentTurn, *, focused: bool) -> str:
        question = DecisionQuestion(instructions=_CONTROL, criteria=_FOCUSED_ACTIONS if focused else _UNFOCUSED_ACTIONS)
        state = {"request": turn.query, "destination": {"name": "notes", "focused": focused}}
        answer = await decide(
            self._decisions, "note-action", "note_action", question, state, participant_id=turn.participant_id
        )
        return answer.choice

    async def _save_note(self, participant_id: str, text: str) -> str:
        """Store a note, share it with general conversation, and return the acknowledgment to speak."""

        note = text.strip()
        self._notes.setdefault(participant_id, []).append(note)
        # The note is saved locally either way; failed sharing only changes the acknowledgment.
        return _NOTED if await self._share(participant_id, note) else _NOT_SHARED

    def _read_back(self, participant_id: str) -> str:
        notes = self._notes.get(participant_id)
        return "Your notes: " + "; ".join(notes) if notes else _NO_NOTES

    async def _share(self, participant_id: str, note: str) -> bool:
        """Publish one saved note; return whether it was shared."""

        if self._publish_note is None:
            return True
        publisher = asyncio.create_task(
            self._publish_note(participant_id, ConversationContextUpdate(text=note)),
            name=f"conversation-example-note-context:{participant_id}",
        )
        # A started publication always finishes, so shared context never misses a saved note.
        try:
            await wait_despite_cancellation({publisher})
        except asyncio.CancelledError:
            if not publisher.cancelled() and (error := publisher.exception()) is not None:
                logger.opt(exception=error).error(
                    "note context publication failed while draining cancellation participant={!r}", participant_id
                )
            raise
        try:
            publisher.result()
        except RuntimeClosedError:
            return False
        except ExceptionGroup:
            # A failed context subscriber; the note is saved, so only the acknowledgment changes.
            logger.exception("note context publication failed participant={!r}", participant_id)
            return False
        return True


__all__ = ["NoteTakingAgent"]
