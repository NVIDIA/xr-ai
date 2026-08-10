# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the sample's focused agent Functions into its top-level workflow."""

from pathlib import Path

from loguru import logger
from nat.builder.function import LambdaFunction
from nat.plugin_api import Builder, Function, FunctionBaseConfig, FunctionInfo, FunctionRef, LLMRef
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from xr_ai_models import LLMService
from xr_ai_nat.functions.text_memory import RecallConversationRequest
from xr_ai_nat.llm import ModelsLLMConfig

from .agents import (
    AppearanceAgentConfig,
    MemoryAgentConfig,
    ObjectAgentConfig,
    PlacementAgentConfig,
    VisionAgentConfig,
)
from .models import SceneReply, SceneRequest
from .scene import SceneContext
from .spatial_ops import CreationLedger

_PROMPT = Path(__file__).with_name("supervisor_prompt.txt")
_LLM_NAME = LLMRef("scene_llm")

# Words that cannot end a complete command; VAD truncation leaves them
# dangling ("Put the sphere on the"). The model reliably autocompletes such
# fragments from scene state, so they never reach it.
_DANGLING_WORDS = frozenset(
    "a an the my your its of to on in at by and or with near under over "
    "onto into between behind above below beside toward towards from".split()
)


def _is_truncated(transcript: str) -> bool:
    words = transcript.strip().rstrip(".?!,;").lower().split()
    return bool(words) and words[-1] in _DANGLING_WORDS


_ARTICLES = frozenset({"a", "an", "the", "my", "your", "its"})

_TRUNCATED_ASK = "I think I missed the end of that."

_CANCEL_PHRASES = frozenset({
    "never mind", "nevermind", "forget it", "forget that", "cancel", "cancel that", "no", "stop",
})

_ACTION_VERBS = frozenset(
    "put place move make create add remove delete drop turn rotate resize double halve shrink "
    "grow recolor paint swap bring push pull raise lower undo change set scoot flip clear".split()
)


def _truncated_reply(transcript: str) -> str:
    words = transcript.strip().rstrip(".?!,;").split()
    tail = words[-1]
    if tail.lower() in _ARTICLES and len(words) >= 2:
        tail = f"{words[-2]} {tail}"
    return f"{_TRUNCATED_ASK} {tail.capitalize()} what?"


def _splice_completion(prefix: str, completion: str) -> str:
    """Join a cut-off request with its answer, merging overlapping words."""
    head = prefix.strip().rstrip(".?!,;")
    tail_words = completion.strip().rstrip(".?!,;").split()
    head_words = head.split()
    for overlap in (3, 2, 1):
        if (
            len(head_words) >= overlap
            and len(tail_words) >= overlap
            and [word.lower() for word in head_words[-overlap:]]
            == [word.lower() for word in tail_words[:overlap]]
        ):
            tail_words = tail_words[overlap:]
            break
    return f"{head} {' '.join(tail_words)}".strip() + "."


def _resolve_truncation_reply(prefix: str, transcript: str) -> str | None:
    """Decide what a turn following the truncation ask-back means.

    Returns the transcript to process, or None when the turn cancels the
    cut-off request. A turn with its own action verb is a fresh command; a
    bare fragment answers the ask and splices onto the cut-off prefix.
    """
    words = transcript.strip().rstrip(".?!,;").lower()
    if words in _CANCEL_PHRASES:
        return None
    if any(word in _ACTION_VERBS for word in words.split()):
        return transcript
    return _splice_completion(prefix, transcript)


class SceneSupervisorConfig(FunctionBaseConfig, name="xr_render_scene_supervisor"):
    """Registry and tracing identity for the supervisor function."""


async def scene_supervisor(
    *,
    builder: Builder,
    llm: LLMService,
    context: SceneContext | None = None,
) -> Function:
    """Compose five subagents without knowing their transitive capabilities."""
    await builder.add_llm(
        _LLM_NAME,
        ModelsLLMConfig(
            service=llm,
            model_name="xr-scene-agent",
            max_tokens=2048,
            temperature=0.0,
            recover_tool_calls=True,
        ),
    )
    if context is None:
        scene_state = await builder.get_function_group("scene_state")
        scene_functions = await scene_state.get_all_functions()
        tracking = await builder.get_function_group("tracking")
        tracking_functions = await tracking.get_all_functions()
        context = SceneContext(
            scene_functions["scene_state__get_scene_state"],
            tracking_functions["tracking__get_user_frame"],
        )
    conversations = await builder.get_function_group("conversations")
    conversation_functions = await conversations.get_all_functions()
    recall = conversation_functions["conversations__recall_conversation"]

    async def recent_conversation(participant_id: str) -> tuple[str, str]:
        """Return the conversation block plus any pending cut-off request."""
        recalled = await recall.ainvoke(RecallConversationRequest(participant_id=participant_id))
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
        lines = [f"  {'User' if entry.role == 'user' else 'Agent'}: {entry.text}" for entry in entries]
        block = (
            "[Recent conversation] (already handled; never a source of new work)\n"
            + "\n".join(lines) + "\n\n"
        )
        return block, pending

    ledger = CreationLedger()
    subagents = (
        ("placement_agent", PlacementAgentConfig(context=context)),
        ("appearance_agent", AppearanceAgentConfig(context=context)),
        ("object_agent", ObjectAgentConfig(context=context, ledger=ledger)),
        ("vision_agent", VisionAgentConfig(context=context)),
        ("memory_agent", MemoryAgentConfig()),
    )
    for name, config in subagents:
        await builder.add_function(name, config)

    reasoning = await builder.add_function(
        "supervisor_reasoning",
        ToolCallAgentWorkflowConfig(
            llm_name=_LLM_NAME,
            tool_names=[FunctionRef(name) for name, _config in subagents],
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
            handle_tool_errors=True,
            max_iterations=12,
            max_empty_response_retries=1,
        ),
    )

    async def supervise(request: SceneRequest) -> SceneReply:
        if _is_truncated(request.transcript):
            return SceneReply(response=_truncated_reply(request.transcript))
        conversation, pending_truncation = await recent_conversation(request.participant_id)
        transcript = request.transcript
        if pending_truncation:
            resolved = _resolve_truncation_reply(pending_truncation, transcript)
            if resolved is None:
                return SceneReply(response="Okay, never mind that.")
            transcript = resolved
        ledger.reset()
        context.take_mutating(request.participant_id)
        context.take_delegated(request.participant_id)
        before = await context.snapshot()
        message = (
            f"Active participant: {request.participant_id}\n"
            f"Utterance timestamp: {request.timestamp_us}\n"
            f"{await context.describe(request.participant_id)}\n\n"
            f"{conversation}"
            f"User request: {transcript}"
        )
        # A failed reasoning pass must degrade, not raise: the verification
        # pass below then gets a chance to complete the turn.
        try:
            output = await reasoning.ainvoke(message, to_type=str)
        except Exception as error:
            logger.error("supervisor reasoning failed: {}", error)
            output = "Something went wrong on my side; please say that again."
        # Conversational turns (no delegation, reply is a question) skip
        # the verification pass so chat stays single-pass. Feeding the diff
        # back on turns that DID mutate makes the model re-delegate
        # completed work with jittered arguments, defeating both scoring
        # and the creation ledger.
        # An ask-back with NO delegation behind it is genuine conversation;
        # an ask-back after a delegation (e.g. vision degraded) is a turn
        # that may still owe a mutation, so it keeps the rescue pass.
        context.take_mutating(request.participant_id)
        delegated_any = context.take_delegated(request.participant_id)
        conversational = not delegated_any and str(output or "").rstrip().endswith("?")
        if not conversational and not SceneContext.changes(before, await context.snapshot()):
            verification = (
                f"{message}\n\n"
                f"Your reply so far: {output}\n"
                "Verified scene changes this turn: none. If the request needed a"
                " scene change, delegate the remaining work now; if it needed"
                " none, repeat your final answer."
            )
            try:
                output = await reasoning.ainvoke(verification, to_type=str)
            except Exception as error:
                logger.error("supervisor verification pass failed: {}", error)
                output = "Something went wrong on my side; please say that again."
        await context.record_moves(request.participant_id, before)
        return SceneReply(response=str(output or "Done."))

    return LambdaFunction.from_info(
        config=SceneSupervisorConfig(),
        info=FunctionInfo.from_fn(
            supervise,
            description="Coordinate focused XR subagents to satisfy one complete request.",
        ),
        instance_name="xr_scene_supervisor",
    )


__all__ = ["scene_supervisor"]
