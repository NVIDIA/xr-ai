# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene supervisor: coordinate five focused subagents over a direct LLM loop."""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService, VLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.text_memory import AddTranscriptRequest, RecallConversationRequest, TextMemoryTools
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.video_memory import VideoMemoryTools
from xr_ai_tools.vision import ImageQueryTool
from xr_render_scene import SceneTools

from ._physical_color import IMAGE_QUERY_SYSTEM_PROMPT, make_physical_color_tool
from ._trace import (
    TurnEvidence,
    current_participant_id,
    current_reference_time_us,
    current_trace_id,
    current_turn_evidence,
)
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

_DANGLING_DETERMINERS = frozenset("a an the my your its".split())

_DANGLING_PREPOSITIONS = frozenset(
    "of to on in at by and or with near under over "
    "onto into between behind above below beside toward towards from".split()
)

_ARTICLES = frozenset({"a", "an", "the", "my", "your", "its"})

# Shared punctuation strip set, unicode ellipsis and quotes included.
# Detection strips per word; the ask-back strips the transcript tail.
_EDGE_PUNCT = ".,!?;:\"'…“”‘’"

_WH_WORDS = frozenset("what which where when why who whom whose how".split())

# Subjects that make a following -ing word a progressive verb ("the wall
# I'm staring at" is complete; "the lamp on the ceiling in" is not).
_PROGRESSIVE_SUBJECTS = frozenset(
    "am is are was were be been being "
    "i'm im he's she's it's you're we're they're user user's".split()
)

_TRUNCATED_ASK = "I think I missed the end of that."

_UNOBSERVED_REPLY = "I haven't been able to observe that, so I can't say."

_CANCEL_PHRASES = frozenset({
    "never mind", "nevermind", "forget it", "forget that", "cancel", "cancel that", "no", "stop",
})

_ACTION_VERBS = frozenset(
    "put place move make create add remove delete drop turn rotate resize double halve shrink "
    "grow recolor paint swap bring push pull raise lower undo change set scoot flip clear "
    "copy duplicate stack tint".split()
)


_MUTATING_AGENTS = frozenset({"placement_agent", "appearance_agent", "object_agent"})

# Seconds to let a scene RPC propagate before diffing snapshots.
_SCENE_SETTLE_S = 0.15


# Leading words that mark a status question about past work ("Did you move
# the cube?"); unlike can/could/will, they cannot open a polite command.
_STATUS_OPENERS = frozenset("did have has had was were".split())


def _wants_mutation(transcript: str) -> bool:
    words = [w.strip(_EDGE_PUNCT) for w in transcript.lower().split()]
    words = [w for w in words if w]
    # A wh- or status question is a query even when it contains an action
    # word ("What color was the first thing I created?", "Did you move it?").
    if not words or words[0] in _WH_WORDS or words[0] in _STATUS_OPENERS:
        return False
    return any(word in _ACTION_VERBS for word in words)


_COMPLETION_CLAIMS = re.compile(
    r"\b(recolou?red|changed|updated|created|added|removed|resized|moved|swapped|"
    r"done|has been|have been|is now|are now|successfully|matched)\b",
    re.IGNORECASE,
)


def _claims_completion(text: str) -> bool:
    return _COMPLETION_CLAIMS.search(text) is not None


def _is_question(text: str) -> bool:
    return text.rstrip().rstrip("\"'”’)").rstrip().endswith("?")


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Hands-and-body claims in our own reply: the only family a camera alone can
# settle, since look-at and point-at answers come from tracking or scene data
# and past tense from memory. Defense in depth over the model's dominant
# register, not a paraphrase parser.
_PERCEPTION_CLAIMS = re.compile(
    r"\byou\s*(?:['’]re|\s+are|\s+(?:appears?|seems?)\s+to\s+be)"
    r"(?:\s+\w+){0,2}\s+(?:holding|carrying|wearing|gripping)\b"
    r"|\bin\s+your(?:\s+\w+){0,2}\s+(?:hands?|grasp|grip)\b",
    re.IGNORECASE,
)

# An honest "I can't see" is the answer the gate wants, so it must survive it.
_PERCEPTION_HEDGES = re.compile(
    r"\bcan(?:['’]t|not)\b|\bcould(?:n['’]t|\s+not)\b|\bunable\b|\bnot\s+sure\b"
    r"|\b(?:do|does|did)(?:n['’]t|\s+not)\s+know\b|\bno\s+camera\b|\bunavailable\b"
    r"|\bhave(?:n['’]t|\s+not)\b",
    re.IGNORECASE,
)


def _claims_perception(text: str) -> bool:
    return any(
        _PERCEPTION_CLAIMS.search(sentence)
        and not _is_question(sentence)
        and not _PERCEPTION_HEDGES.search(sentence)
        for sentence in _SENTENCE_SPLIT.split(text)
    )


def _is_truncated(transcript: str) -> bool:
    words = [w.strip(_EDGE_PUNCT) for w in transcript.strip().lower().split()]
    words = [w for w in words if w]
    if not words:
        return False
    if words[-1] in _DANGLING_DETERMINERS:
        return True
    if words[-1] in _DANGLING_PREPOSITIONS:
        # Only a recognized complete tail construction exempts a trailing
        # preposition: a leading wh-question ("What am I looking at") or a
        # progressive construction right before it ("the wall I'm staring
        # at"). Terminal punctuation, embedded wh-words, and bare -ing nouns
        # ("the ceiling in") all appear on cut input.
        if words[0] in _WH_WORDS:
            return False
        prev = words[-2] if len(words) >= 2 else ""
        subject = words[-3] if len(words) >= 3 else ""
        if prev.endswith("ing") and subject in _PROGRESSIVE_SUBJECTS:
            return False
        return True
    return False


def _truncated_reply(transcript: str) -> str:
    words = transcript.strip().rstrip(_EDGE_PUNCT).split()
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
        vlm: VLMService | None = None,
        images: ImageRegistry | None = None,
        current_frame: CurrentFrameTool | None = None,
        video: VideoMemoryTools | None = None,
        *,
        subagent_tools: list[Tool] | None = None,
    ) -> None:
        context = SceneContext(scene, tracking)
        if subagent_tools is None:
            if vlm is None or images is None or current_frame is None:
                raise ValueError(
                    "vlm, images, and current_frame are required when subagent_tools is not provided"
                )
            image_query = ImageQueryTool(
                images=images,
                vlm=vlm,
                system_prompt="Answer directly from the visible camera image in one short plain-English sentence.",
            )

            physical_color = make_physical_color_tool(
                current_frame,
                ImageQueryTool(images=images, vlm=vlm, system_prompt=IMAGE_QUERY_SYSTEM_PROMPT),
            )

            subagent_tools = [
                make_placement_agent(llm, scene, tracking, context),
                make_appearance_agent(llm, scene, context, physical_color),
                make_object_agent(llm, scene, tracking, context, physical_color),
                make_vision_agent(llm, current_frame, image_query, context, video),
                make_memory_agent(llm, text_memory),
            ]
        self._llm = llm
        self._context = context
        self._text_memory = text_memory
        self._toolset = ToolSet(subagent_tools)
        self._prompt = _PROMPT.read_text(encoding="utf-8").strip()
        self._participant_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._scene_lock: asyncio.Lock = asyncio.Lock()

    def forget_participant(self, participant_id: str) -> None:
        """Drop per-participant state after departure; a reconnecting id starts clean."""
        self._participant_locks.pop(participant_id, None)
        self._context.forget_participant(participant_id)

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

    async def _persist_turn(self, request: SceneRequest, user_text: str, reply_text: str) -> None:
        await self._text_memory.add_transcript.execute(
            AddTranscriptRequest(
                source_id=f"{request.participant_id}:user",
                timestamp_us=request.timestamp_us,
                text=user_text,
            )
        )
        await self._text_memory.add_transcript.execute(
            AddTranscriptRequest(
                source_id=f"{request.participant_id}:agent",
                timestamp_us=time.time_ns() // 1_000,
                text=reply_text,
            )
        )

    async def handle(self, request: SceneRequest) -> SceneReply:
        current_trace_id.set(request.trace_id)
        current_participant_id.set(request.participant_id)
        current_reference_time_us.set(request.timestamp_us)
        if _is_truncated(request.transcript):
            # The ask must reach memory: the next turn's completion splice
            # keys off the recalled truncated-ask reply.
            reply = _truncated_reply(request.transcript)
            await self._persist_turn(request, request.transcript, reply)
            return SceneReply(response=reply)

        async with self._participant_locks[request.participant_id]:
            return await self._handle(request)

    async def _handle(self, request: SceneRequest) -> SceneReply:
        logger.debug(
            "supervisor turn participant={} trace={} transcript={!r}",
            request.participant_id, request.trace_id, request.transcript[:80],
        )
        conversation, pending_truncation = await self._recent_conversation(request.participant_id)
        transcript = request.transcript
        if pending_truncation:
            resolved = _resolve_truncation_reply(pending_truncation, transcript)
            if resolved is None:
                reply = "Okay, never mind that."
                await self._persist_turn(request, request.transcript, reply)
                return SceneReply(response=reply)
            transcript = resolved

        async with self._scene_lock:
            return await self._handle_scene(request, transcript, conversation)

    async def _handle_scene(
        self, request: SceneRequest, transcript: str, conversation: str
    ) -> SceneReply:
        evidence = TurnEvidence()
        current_turn_evidence.set(evidence)
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

        async def _call_model(model_transcript, definitions):
            return await self._llm.chat(
                model_transcript, tools=list(definitions) or None, max_tokens=2048, temperature=0.0
            )

        try:
            result = await run_tool_loop(messages, self._toolset, _call_model, max_iterations=12)
        except ToolLoopError as exc:
            logger.warning("supervisor loop failed ({})", exc)
            reply = "I'm sorry — something went wrong. Please try again."
            await self._persist_turn(request, transcript, reply)
            return SceneReply(response=reply)
        output = result.content

        # Verify only turns with actual mutation intent: a mutating subagent
        # was delegated, or the utterance itself requests a change (which
        # also catches mixed requests where only vision or memory ran).
        delegated = {record.call.name for record in result.tool_calls}
        needs_verification = bool(delegated & _MUTATING_AGENTS) or _wants_mutation(transcript)

        await asyncio.sleep(_SCENE_SETTLE_S)
        latest = result
        verify_no_change = needs_verification and not SceneContext.changes(
            before, await self._context.snapshot()
        )
        if verify_no_change:
            if delegated & _MUTATING_AGENTS:
                nudge = (
                    "Verified scene changes this turn: none. If the request needed a"
                    " scene change, delegate the remaining work now; if it needed"
                    " none, repeat your final answer."
                )
            else:
                # No scene-changing subagent ran, so a repeat-answer offer
                # would invite an unsupported completion claim.
                nudge = (
                    "Verified scene changes this turn: none, and no scene-changing"
                    " subagent was used. The request asks for a change: delegate the"
                    " remaining work to the right subagent now. If the change cannot"
                    " be done, say plainly that nothing was changed and why; never"
                    " reply that a change was made."
                )
            verification_messages = list(latest.messages) + [
                ChatMessage(role="user", content=nudge),
            ]
            try:
                result2 = await run_tool_loop(
                    verification_messages, self._toolset, _call_model, max_iterations=6
                )
            except ToolLoopError as exc:
                logger.warning("supervisor verification failed ({})", exc)
            else:
                output = result2.content
                latest = result2
            await asyncio.sleep(_SCENE_SETTLE_S)

        perception_retried = False
        if output and _claims_perception(output) and evidence.observed == 0:
            logger.warning("perception claim without observation: {!r}", output[:80])
            perception_retried = True
            perception_nudge = (
                "Your reply asserts what the user is holding, carrying, or"
                " wearing, but no camera observation was made this turn."
                " Delegate vision_agent now with that exact question and"
                " answer only from its result; if it cannot see, say so."
            )
            perception_messages = list(latest.messages) + [
                ChatMessage(role="user", content=perception_nudge),
            ]
            try:
                perception_result = await run_tool_loop(
                    perception_messages, self._toolset, _call_model, max_iterations=6
                )
            except ToolLoopError as exc:
                logger.warning("perception retry failed ({})", exc)
                output = ""
            else:
                output = perception_result.content
            await asyncio.sleep(_SCENE_SETTLE_S)

        # One text policy over the final reply, whichever loop produced it.
        # Perception replaces before the mutation gate appends, so a turn that
        # trips both keeps the no-change fact.
        if evidence.observed == 0 and (
            (perception_retried and not output) or (output and _claims_perception(output))
        ):
            output = _UNOBSERVED_REPLY
        if verify_no_change:
            # Success is evidence-backed: a completion claim may stand only
            # when a scene write was applied (or the requested state already
            # held), never on the model's wording alone. A claim-free
            # explanation keeps its why; the no-change fact is appended.
            no_evidence = (
                evidence.applied == 0
                and evidence.satisfied == 0
                and not SceneContext.changes(before, await self._context.snapshot())
            )
            if no_evidence and (not output or _claims_completion(output)):
                logger.warning("no mutation evidence; replacing reply {!r}", (output or "")[:80])
                output = "I couldn't make that change; nothing in the scene was changed."
            elif no_evidence and not _is_question(output):
                output = output.rstrip() + " Nothing in the scene was changed."

        await self._context.record_moves(request.participant_id, before)
        reply_text = output or "Done."
        await self._persist_turn(request, transcript, reply_text)
        return SceneReply(response=reply_text)


__all__ = ["SceneSupervisor"]
