# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own participant-authored notes as a focused route destination."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_conversation import Conversation, ConversationContextUpdate
from xr_ai_interaction import ResponseOutput
from xr_ai_models import DecisionQuestion, DecisionService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutableAgent, RoutingAid
from xr_ai_runtime import AgentRuntime, RuntimeClosedError, Topic

ContextPublisher = Callable[[str, ConversationContextUpdate], Awaitable[None]]
_CONTROL = (Path(__file__).parent / "prompts/note_control.txt").read_text(encoding="utf-8").strip()


class NoteTakingAgent(RoutableAgent):
    """Keep participant-authored notes in an explicit, focused session.

    Admission classifies the current request and focus mode, independently of
    saved content. Save acknowledgments follow successful local storage.
    """

    def __init__(
        self,
        conversation: Conversation,
        decisions: DecisionService,
        *,
        runtime: AgentRuntime,
        output_topic: Topic[ResponseOutput],
        context_budget_bytes: int,
        publish_context: ContextPublisher | None = None,
    ) -> None:
        if context_budget_bytes <= 0:
            raise ValueError("context_budget_bytes must be positive")
        self._conversation = conversation
        self._static_description = (
            "A participant-scoped note-taking destination. It starts only when explicitly requested, captures "
            "user-authored notes, and stays focused until the participant finishes, including across a question "
            "answered by another destination. It does not "
            "provide general knowledge or inspect the current camera."
        )
        self._context_budget_bytes = context_budget_bytes
        self._publish_context = publish_context
        self._runtime = runtime
        self._output_topic = output_topic
        self._decisions = decisions
        self._pending: dict[str, tuple[str, str, bool]] = {}
        self._notes: dict[str, list[str]] = {}
        self._active_participants: set[str] = set()

    # RoutableAgent contract: note-specific routing and lifecycle behavior.
    def routing_aid(self, participant_id: str) -> RoutingAid:
        notes = self._notes.get(participant_id, ())
        description = self._static_description
        if notes:
            cue = f"Latest saved note (context only; the note session may be closed): {notes[-1]}"
            candidate = f"{description}\n{cue}"
            if len(candidate.encode("utf-8")) <= self._context_budget_bytes:
                description = candidate
        return RoutingAid(name="notes", description=description)

    async def participant_joined(self, participant_id: str) -> None:
        """Publish the note capability and initial participant-specific state."""

        await self._publish_context_snapshot(participant_id)

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        # Admission chooses a current action; saved content belongs to readback and shared context.
        state = {
            "request": turn.query,
            "destination": {"name": "notes", "focused": focused},
        }
        criteria = {
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
        if not focused:
            criteria.update(
                start="The participant explicitly wants to begin a note-taking session.",
                other="The current request is not a note action or an explicit request to save content.",
            )
        with nemo_relay.scope.scope(
            "conversation-example.note-action",
            nemo_relay.ScopeType.Agent,
            input=state,
            metadata={"participant_id": turn.participant_id},
        ) as scope:
            response = await self._decisions.decide(
                state,
                {"note_action": DecisionQuestion(instructions=_CONTROL, criteria=criteria)},
            )
            action = response.answers["note_action"].choice
            nemo_relay.scope.event(
                "conversation-example.note-action-result",
                handle=scope,
                data={"choice": action, "confidence": response.answers["note_action"].confidence},
                metadata={"participant_id": turn.participant_id},
            )
        if action not in criteria:
            raise ValueError(f"Clef returned invalid note action {action!r}")

        if not focused and action not in {"start", "capture"}:
            return Admission("decline")
        if action == "handoff":
            return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, action, focused)
        try:
            return await self._conversation.admit(turn, focused=focused)
        except BaseException:
            # Clear provisional action state before propagating failure or cancellation.
            self._pending.pop(turn.participant_id, None)
            raise

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[bool]]) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)
        if pending is None or pending[0] != turn.turn_id:
            return DestinationResult("answer", "")
        action, was_focused = pending[1:]

        async def emit(text: str) -> None:
            interrupt = await started()
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError
            await self._runtime.publish(
                self._output_topic,
                ResponseOutput(
                    text=text,
                    response_id=turn.turn_id,
                    interrupt=interrupt,
                    timestamp_us=turn.timestamp_us,
                ),
                participant_id=turn.participant_id,
                source="notes",
            )

        return await self._answer_control(turn, action, was_focused, emit)

    async def _answer_control(
        self,
        turn: AgentTurn,
        action: str,
        was_focused: bool,
        emit: Callable[[str], Awaitable[None]],
    ) -> DestinationResult:
        if action == "capture":
            was_active = turn.participant_id in self._active_participants
            self._active_participants.add(turn.participant_id)
            try:
                accepted = self._append_note(turn.participant_id, turn.query)
                if not accepted:
                    await self._publish_note_status(turn.participant_id)
                    text = "That note is too long for the available note space, so I didn't save it."
                    await emit(text)
                    return DestinationResult("focus", text)
                try:
                    await self._publish_context_snapshot(turn.participant_id)
                except asyncio.CancelledError:
                    raise
                except RuntimeClosedError:
                    # ignored: the note is saved locally; shutdown only prevents optional sharing.
                    text = "I saved that note here, but couldn't share it with general conversation."
                    await emit(text)
                    return DestinationResult("focus", text)
                except ExceptionGroup:
                    logger.exception("note context publication failed participant={!r}", turn.participant_id)
                    text = "I saved that note here, but couldn't share it with general conversation."
                    await emit(text)
                    return DestinationResult("focus", text)
                text = "Noted."
                await emit(text)
                return DestinationResult("focus", text)
            except BaseException:
                await self._restore_note_activity(turn.participant_id, was_active)
                raise
        if action in ("start", "finish"):
            was_active = turn.participant_id in self._active_participants
            starting = action == "start"
            if starting:
                self._active_participants.add(turn.participant_id)
                text = "I’m ready to take notes."
            else:
                self._active_participants.discard(turn.participant_id)
                text = "Note-taking is finished."
            try:
                await self._publish_note_status(turn.participant_id)
                await emit(text)
                return DestinationResult("focus" if starting else "release", text)
            except BaseException:
                await self._restore_note_activity(turn.participant_id, was_active)
                raise
        if action == "readback":
            notes = self._notes.get(turn.participant_id, ())
            text = "Your notes: " + "; ".join(notes) if notes else "There are no notes in this session yet."
            await emit(text)
            return DestinationResult("focus" if was_focused else "answer", text)
        raise ValueError(f"invalid note action {action!r}")

    async def release(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)
        if participant_id in self._active_participants:
            self._active_participants.discard(participant_id)
            await self._publish_note_status(participant_id)

    async def participant_left(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)
        self._notes.pop(participant_id, None)
        self._active_participants.discard(participant_id)
        await self._conversation.participant_left(participant_id)

    # Private note storage and context publication helpers.
    def _append_note(self, participant_id: str, text: str) -> bool:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        note = f"{timestamp} — {text.strip()}"
        if len(note.encode("utf-8")) > self._context_budget_bytes:
            return False
        notes = self._notes.setdefault(participant_id, [])
        notes.append(note)
        while notes and len("\n".join(notes).encode("utf-8")) > self._context_budget_bytes:
            notes.pop(0)
        return True

    async def _publish_context_snapshot(self, participant_id: str) -> None:
        if self._publish_context is None:
            return
        notes = self._notes.get(participant_id, ())
        state = "active" if participant_id in self._active_participants else "not active"
        text = (
            "Capability: capture and review participant-authored notes when explicitly requested. "
            f"Session status: {state}.\n"
            + ("Saved notes:\n" + "\n".join(notes) if notes else "No notes have been saved in this session.")
        )
        publisher = asyncio.create_task(
            self._publish_context(participant_id, ConversationContextUpdate(text=text)),
            name=f"conversation-example-note-context:{participant_id}",
        )
        try:
            await asyncio.shield(publisher)
        except asyncio.CancelledError:
            while not publisher.done():
                try:
                    await asyncio.shield(publisher)
                except asyncio.CancelledError:
                    continue
                except Exception:  # noqa: BLE001 -- drain the child failure without masking caller cancellation.
                    break
            if not publisher.cancelled():
                error = publisher.exception()
                if error is not None:
                    logger.opt(exception=error).error(
                        "note context publication failed while draining cancellation participant={!r}",
                        participant_id,
                    )
            raise

    async def _publish_note_status(self, participant_id: str) -> None:
        """Keep optional status sharing from interrupting note-session control."""

        try:
            await self._publish_context_snapshot(participant_id)
        except RuntimeClosedError:
            return
        except ExceptionGroup:
            logger.exception("note status context publication failed participant={!r}", participant_id)

    async def _restore_note_activity(self, participant_id: str, was_active: bool) -> None:
        """Restore pre-turn activity when an action cannot commit its disposition."""

        if was_active:
            self._active_participants.add(participant_id)
        else:
            self._active_participants.discard(participant_id)
        try:
            await self._publish_context_snapshot(participant_id)
        except asyncio.CancelledError:
            # Snapshot publication drains its child before propagating cancellation.
            pass
        except Exception:  # noqa: BLE001 -- rollback must not replace the action's original failure.
            logger.exception("note activity rollback publication failed participant={!r}", participant_id)


__all__ = ["NoteTakingAgent"]
