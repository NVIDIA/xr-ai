# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose shared SDK primitives into the conversation example worker.

Only the worker entry point, sample evaluator, and repository tests import this
module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from xr_ai_conversation import (
    ContextManager,
    Conversation,
    ConversationContextUpdate,
)
from xr_ai_logging import setup_logging
from xr_ai_models import (
    DecisionService,
    LLMService,
    ModelsConfig,
    TTSService,
    VLMService,
    load_models_config,
    make_decision,
    make_llm,
    make_tts,
    make_vlm,
)
from xr_ai_routing import AgentRouter, RoutingAid
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import StreamingImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import VoiceGateConfig, load_voice_gate_config

from ._alert_agent import ALERT_CAPABILITY_TOPIC, VisualAlertAgent
from ._config import WorkerConfig
from ._note_taking import NoteTakingAgent
from ._topics import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from ._visual_agents import OBSERVATION_CONTEXT_TOPIC, VisualObservationAgent

_CONVERSATION_SCOPE = (
    "Provide text assistance across subjects and recall relevant conversation history and participant-authored "
    "notes. "
    "If answering depends on what the camera shows now, use direct observation even when prior "
    "turns contain the same or a similar question and answer. "
    "This destination does not inspect the camera."
)
_PROMPTS = Path(__file__).parent / "prompts"
_CONVERSATION_INSTRUCTIONS = (_PROMPTS / "conversation.txt").read_text(encoding="utf-8").strip()
_NOTES_INSTRUCTIONS = (_PROMPTS / "note_taking.txt").read_text(encoding="utf-8").strip()
_NOTES_SCOPE = (
    "An explicitly started note-taking session for capturing and reviewing participant-authored notes. "
    "It does not provide general knowledge or inspect current-camera content."
)
_NOTES_CONTEXT_TOPIC = Topic("conversation-example.foreground-notes", ConversationContextUpdate)
ParticipantStep = Callable[[str], Awaitable[None]]


def _acknowledgment_models(models: ModelsConfig) -> ModelsConfig:
    """Clone the configured LLM role with reasoning disabled for brief acks."""

    spec = models.llm("llm")
    extras = dict(spec.adapter.default_extras)
    extras.pop("max_tokens", None)
    template = dict(extras.get("chat_template_kwargs", {}))
    template.pop("reasoning_budget", None)
    template.pop("thinking_budget", None)
    if template:
        extras["chat_template_kwargs"] = template
    else:
        extras.pop("chat_template_kwargs", None)
    thinking = replace(spec.adapter.thinking, default=False)
    acknowledgment_spec = replace(
        spec,
        adapter=replace(spec.adapter, default_extras=extras, thinking=thinking),
    )
    return ModelsConfig({**models.entries, "acknowledgment_llm": acknowledgment_spec})


@asynccontextmanager
async def _relay_event_log(log_file: Path) -> AsyncIterator[Path]:
    event_path = log_file.parent / "relay-events.jsonl"
    lock = Lock()
    subscriber = "conversation-example-compact-event-log"

    with event_path.open("w", encoding="utf-8") as sink:

        def write_event(event: nemo_relay.Event) -> None:
            if event.kind == "mark" and event.name == "llm.chunk":
                return
            with lock:
                sink.write(event.to_json())
                sink.write("\n")
                sink.flush()

        nemo_relay.subscribers.register(subscriber, write_event)
        try:
            yield event_path
        finally:
            await nemo_relay.subscribers.flush_async()
            nemo_relay.subscribers.deregister(subscriber)


async def run_app(
    config: WorkerConfig,
    *,
    ready_file: Path | None = None,
) -> None:
    """Compose services and destinations, then run until the voice session ends."""

    log_file = setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)

    # This composition root owns shared clients and closes them after the runtime.
    llm = make_llm(models, "llm")
    acknowledgment_llm = (
        make_llm(_acknowledgment_models(models), "acknowledgment_llm")
        if config.acknowledgment_delay_secs is not None
        else None
    )
    decisions: DecisionService = make_decision(models, "router")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")
    transport = HubVoiceTransport()
    voice = _voice_agent(
        config, models, voice_gate, transport, tts=tts, llm=llm, decisions=decisions, vlm=vlm, ready_file=ready_file
    )

    # Current images remain on the hub path; the worker receives image references.
    runtime = AgentRuntime()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=transport.endpoint,
        images=images,
        frame_max_age_s=config.frame_max_age_s,
        frame_timeout_s=config.frame_timeout_s,
    )

    # General answers and focused notes use distinct instructions and their own context;
    # AgentRouter supplies shared turn history. This application owns both context managers.
    context, notes_context = _context_managers(config, llm)
    conversation = Conversation(
        llm,
        context=context,
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid(name="conversation", description=_CONVERSATION_SCOPE),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    notes_conversation = Conversation(
        llm,
        context=notes_context,
        instructions=_NOTES_INSTRUCTIONS,
        routing=RoutingAid(name="notes", description=_NOTES_SCOPE),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )

    # Each destination owns its tools, admission policy, and sample state.
    observation = VisualObservationAgent(
        frames,
        StreamingImageQueryTool(images=images, vlm=vlm, system_prompt=config.system_prompt),
        _context_publisher(runtime, OBSERVATION_CONTEXT_TOPIC),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    notes = NoteTakingAgent(
        notes_conversation,
        decisions,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        context_budget_bytes=config.context_budget_bytes,
        publish_context=_context_publisher(runtime, _NOTES_CONTEXT_TOPIC),
    )
    alerts = VisualAlertAgent(
        frames,
        images,
        decisions,
        _context_publisher(runtime, ALERT_CAPABILITY_TOPIC),
        vlm=vlm,
        llm=llm,
        context=context,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        check_interval_secs=config.alert_check_interval_secs,
        confirm_interval_secs=config.alert_confirm_interval_secs,
    )

    async def release_frames(participant_id: str) -> None:
        frames.release(participant_id)

    router = AgentRouter(
        {"conversation": conversation, "observation": observation, "alerts": alerts, "notes": notes},
        decisions,
        acknowledgment_llm=acknowledgment_llm,
        acknowledgment_delay_secs=config.acknowledgment_delay_secs,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        initialize_participant=_in_order(
            notes.participant_joined, alerts.participant_joined, observation.participant_joined
        ),
        # Routes used for turns are also cleaned by the router. These idempotent hooks
        # cover resources initialized at join but never selected for a turn.
        cleanup_participant=_all_of(
            notes.participant_left, alerts.participant_left, context.clear, notes_context.clear, release_frames
        ),
        set_status=transport.endpoint.set_status,
    )
    # Only general conversation consumes the configured shared context sources.
    if config.summary_context_topics or config.incremental_context_topics:
        runtime.register("conversation-context", context.subscriber())
    runtime.register("conversation-example", router.runtime_subscriber())
    runtime.register("voice", voice)

    logger.info("Relay events → {}", log_file.parent / "relay-events.jsonl")
    logger.info("conversation-example starting")
    try:
        async with _relay_event_log(log_file):
            async with runtime:
                try:
                    await voice.run(runtime)
                finally:
                    await router.stop()
    finally:
        await conversation.close()
        await notes_conversation.close()
        await context.close()
        await notes_context.close()
        await alerts.stop()
        if acknowledgment_llm is not None:
            await acknowledgment_llm.close()
        await llm.close()
    logger.info("conversation-example stopped")


def _voice_agent(
    config: WorkerConfig,
    models: ModelsConfig,
    voice_gate: VoiceGateConfig,
    transport: HubVoiceTransport,
    *,
    tts: TTSService,
    llm: LLMService,
    decisions: DecisionService,
    vlm: VLMService,
    ready_file: Path | None,
) -> VoiceAgent:
    """Build the voice front end; it owns transport, speech handling, readiness, and voice cleanup."""

    return VoiceAgent.from_models(
        query_topic=USER_QUERY_TOPIC,
        models=models,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=voice_gate,
        probes={"llm": llm.health, "router": decisions.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(vlm, decisions),
        text_topic="vlm.response",
        idle_timeout_secs=config.idle_timeout_secs,
        transport=transport,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )


def _context_managers(config: WorkerConfig, llm: LLMService) -> tuple[ContextManager, ContextManager]:
    """Return general context, which reads the configured topics, and private note-session context."""

    sources = (*config.summary_context_topics, *config.incremental_context_topics)
    # Dialogue and each context topic get an equal share of the budget.
    section_budget_bytes = config.context_budget_bytes // (len(sources) + 1)
    if section_budget_bytes <= 0:
        raise ValueError("context_budget_bytes must provide a positive budget for dialogue and each context topic")
    general = ContextManager(
        llm,
        summary_context_topics=config.summary_context_topics,
        incremental_context_topics=config.incremental_context_topics,
        dialogue_budget_bytes=section_budget_bytes,
        topic_budget_bytes={name: section_budget_bytes for name in sources},
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )
    notes = ContextManager(
        llm,
        dialogue_budget_bytes=config.context_budget_bytes,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )
    return general, notes


def _context_publisher(
    runtime: AgentRuntime, topic: Topic[ConversationContextUpdate]
) -> Callable[[str, ConversationContextUpdate], Awaitable[None]]:
    async def publish(participant_id: str, update: ConversationContextUpdate) -> None:
        await runtime.publish(topic, update, participant_id=participant_id)

    return publish


def _in_order(*steps: ParticipantStep) -> ParticipantStep:
    async def run(participant_id: str) -> None:
        for step in steps:
            await step(participant_id)

    return run


def _all_of(*steps: ParticipantStep) -> ParticipantStep:
    """Run every step even when one fails, then raise the failures together."""

    async def run(participant_id: str) -> None:
        errors: list[Exception] = []
        for step in steps:
            try:
                await step(participant_id)
            except Exception as error:  # noqa: BLE001 -- retain independent cleanup failures.
                errors.append(error)
        if errors:
            raise ExceptionGroup(f"failed to clean up participant {participant_id!r}", errors)

    return run
