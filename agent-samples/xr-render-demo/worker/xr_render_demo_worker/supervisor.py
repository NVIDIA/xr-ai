# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene supervisor: coordinate five focused subagents over a direct LLM loop."""

from __future__ import annotations

from pathlib import Path

from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.historical_vision import HistoricalVisionTool
from xr_ai_tools.live_vision import LiveVisionTool
from xr_ai_tools.text_memory import RecallConversationRequest, TextMemoryTools
from xr_ai_tools.tool_calling import tool_definitions
from xr_ai_tools.tracking import TrackingTools
from xr_render_scene import SceneTools

from ._loop import tool_loop
from .agents import (
    make_appearance_agent,
    make_memory_agent,
    make_object_agent,
    make_placement_agent,
    make_vision_agent,
)
from .models import SceneReply, SceneRequest
from .scene import SceneContext

_PROMPT = Path(__file__).with_name("supervisor_prompt.txt")

_DANGLING_WORDS = frozenset(
    "a an the my your its of to on in at by and or with near under over "
    "onto into between behind above below beside toward towards from".split()
)

_ARTICLES = frozenset({"a", "an", "the", "my", "your", "its"})

_TRUNCATED_ASK = "I think I missed the end of that."

_CANCEL_PHRASES = frozenset({
    "never mind", "nevermind", "forget it", "forget that", "cancel", "cancel that", "no", "stop",
})

_ACTION_VERBS = frozenset(
    "put place move make create add remove delete drop turn rotate resize double halve shrink "
    "grow recolor paint swap bring push pull raise lower undo change set scoot flip clear".split()
)


def _is_truncated(transcript: str) -> bool:
    words = transcript.strip().rstrip(".?!,;").lower().split()
    return bool(words) and words[-1] in _DANGLING_WORDS


def _truncated_reply(transcript: str) -> str:
    words = transcript.strip().rstrip(".?!,;").split()
    tail = words[-1]
    if tail.lower() in _ARTICLES and len(words) >= 2:
        tail = f"{words[-2]} {tail}"
    return f"{_TRUNCATED_ASK} {tail.capitalize()} what?"


def _splice_completion(prefix: str, completion: str) -> str:
    head = prefix.strip().rstrip(".?!,;")
    tail_words = completion.strip().rstrip(".?!,;").split()
    head_words = head.split()
    for overlap in (3, 2, 1):
        if (
            len(head_words) >= overlap
            and len(tail_words) >= overlap
            and [w.lower() for w in head_words[-overlap:]] == [w.lower() for w in tail_words[:overlap]]
        ):
            tail_words = tail_words[overlap:]
            break
    return f"{head} {' '.join(tail_words)}".strip() + "."


def _resolve_truncation_reply(prefix: str, transcript: str) -> str | None:
    words = transcript.strip().rstrip(".?!,;").lower()
    if words in _CANCEL_PHRASES:
        return None
    if any(word in _ACTION_VERBS for word in words.split()):
        return transcript
    return _splice_completion(prefix, transcript)


class SceneSupervisor:

    def __init__(
        self,
        llm: LLMService,
        scene: SceneTools,
        tracking: TrackingTools,
        text_memory: TextMemoryTools,
        live_vision: LiveVisionTool | None = None,
        past_vision: HistoricalVisionTool | None = None,
        *,
        subagent_tools: list[Tool] | None = None,
    ) -> None:
        context = SceneContext(scene, tracking)
        if subagent_tools is None:
            if live_vision is None or past_vision is None:
                raise ValueError("live_vision and past_vision are required when subagent_tools is not provided")
            subagent_tools = [
                make_placement_agent(llm, scene, tracking, context),
                make_appearance_agent(llm, scene, context),
                make_object_agent(llm, scene, tracking, context),
                make_vision_agent(llm, live_vision, past_vision, context),
                make_memory_agent(llm, text_memory),
            ]
        self._llm = llm
        self._context = context
        self._text_memory = text_memory
        self._toolset = ToolSet(subagent_tools)
        self._tool_defs = tool_definitions(self._toolset)
        self._prompt = _PROMPT.read_text(encoding="utf-8").strip()

    async def _recent_conversation(self, participant_id: str) -> tuple[str, str]:
        recalled = await self._text_memory.recall_conversation.execute(
            RecallConversationRequest(participant_id=participant_id)
        )
        entries = recalled.entries[-8:]
        if not entries:
            return "", ""
        pending = ""
        if (
            len(entries) >= 2
            and entries[-1].role == "agent"
            and entries[-1].text.startswith(_TRUNCATED_ASK)
            and entries[-2].role == "user"
        ):
            pending = entries[-2].text
        lines = [f"  {'User' if e.role == 'user' else 'Agent'}: {e.text}" for e in entries]
        block = (
            "[Recent conversation] (already handled; never a source of new work)\n"
            + "\n".join(lines) + "\n\n"
        )
        return block, pending

    async def handle(self, request: SceneRequest) -> SceneReply:
        if _is_truncated(request.transcript):
            return SceneReply(response=_truncated_reply(request.transcript))

        conversation, pending_truncation = await self._recent_conversation(request.participant_id)
        transcript = request.transcript
        if pending_truncation:
            resolved = _resolve_truncation_reply(pending_truncation, transcript)
            if resolved is None:
                return SceneReply(response="Okay, never mind that.")
            transcript = resolved

        self._context.take_mutating(request.participant_id)
        self._context.take_delegated(request.participant_id)
        before = await self._context.snapshot()

        user_message = (
            f"Active participant: {request.participant_id}\n"
            f"Utterance timestamp: {request.timestamp_us}\n"
            f"{await self._context.describe(request.participant_id)}\n\n"
            f"{conversation}"
            f"User request: {transcript}"
        )
        messages = [
            ChatMessage(role="system", content=self._prompt),
            ChatMessage(role="user", content=user_message),
        ]

        output = await tool_loop(
            self._llm, messages, self._tool_defs, self._toolset, max_tokens=2048
        )

        # Conversational turns (no delegation, reply is a question) skip
        # the verification pass. An ask-back after delegation keeps the pass
        # in case the subagent owed a mutation it didn't complete.
        self._context.take_mutating(request.participant_id)
        delegated_any = self._context.take_delegated(request.participant_id)
        conversational = not delegated_any and str(output or "").rstrip().endswith("?")

        if not conversational and not SceneContext.changes(before, await self._context.snapshot()):
            verification_messages = messages + [
                ChatMessage(role="assistant", content=output or ""),
                ChatMessage(role="user", content=(
                    "Verified scene changes this turn: none. If the request needed a"
                    " scene change, delegate the remaining work now; if it needed"
                    " none, repeat your final answer."
                )),
            ]
            output = await tool_loop(
                self._llm, verification_messages, self._tool_defs, self._toolset, max_tokens=2048
            )

        await self._context.record_moves(request.participant_id, before)
        return SceneReply(response=output or "Done.")


__all__ = ["SceneSupervisor"]
