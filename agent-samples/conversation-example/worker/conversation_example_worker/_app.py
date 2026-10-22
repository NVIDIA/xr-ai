# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose shared SDK primitives into the conversation example worker.

Only the worker entry point, sample evaluator, and repository tests import this
module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from xr_ai_conversation import (
    Conversation,
    ConversationContextUpdate,
)
from xr_ai_logging import setup_logging
from xr_ai_models import (
    DecisionService,
    ModelsConfig,
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
from xr_ai_tools.vision import ImageQueryTool, StreamingImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from ._config import WorkerConfig
from ._monitoring import MONITOR_WINDOW_TOPIC, VisualMonitorAgent
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
    "notes. Use active monitor observations only when the current request is covered by the participant's active "
    "monitor task. If answering depends on what the camera shows now, use direct observation even when prior "
    "turns contain the same or a similar question and answer. Historical monitor records are not current-camera "
    "evidence. This destination does not inspect the camera."
)
_PROMPTS = Path(__file__).parent / "prompts"
_CONVERSATION_INSTRUCTIONS = (_PROMPTS / "conversation.txt").read_text(encoding="utf-8").strip()
_NOTES_INSTRUCTIONS = (_PROMPTS / "note_taking.txt").read_text(encoding="utf-8").strip()
_MONITOR_VLM_INSTRUCTIONS = (_PROMPTS / "monitor_observation.txt").read_text(encoding="utf-8").strip()
_NOTES_CONTEXT_TOPIC = Topic("conversation-example.foreground-notes", ConversationContextUpdate)


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

    # VoiceAgent owns transport, speech handling, readiness, and voice cleanup.
    transport = HubVoiceTransport()
    voice = VoiceAgent.from_models(
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

    # Current images remain on the hub path; the worker receives image references.
    runtime = AgentRuntime()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=transport.endpoint,
        images=images,
        frame_max_age_s=config.frame_max_age_s,
        frame_timeout_s=config.frame_timeout_s,
    )
    vision = StreamingImageQueryTool(
        images=images,
        vlm=vlm,
        system_prompt=config.system_prompt,
    )
    context_sources = (*config.summary_context_topics, *config.incremental_context_topics)
    section_budget_bytes = config.context_budget_bytes // (len(context_sources) + 1)
    if section_budget_bytes <= 0:
        raise ValueError("context_budget_bytes must provide a positive budget for dialogue and each context topic")
    topic_budgets = {name: section_budget_bytes for name in context_sources}
    # General answers and focused notes use distinct instructions; AgentRouter supplies shared turn history.
    conversation = Conversation(
        llm,
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid(name="conversation", description=_CONVERSATION_SCOPE),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        summary_context_topics=config.summary_context_topics,
        incremental_context_topics=config.incremental_context_topics,
        dialogue_budget_bytes=section_budget_bytes,
        topic_budget_bytes=topic_budgets,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )
    notes_conversation = Conversation(
        llm,
        instructions=_NOTES_INSTRUCTIONS,
        routing=RoutingAid(
            name="notes",
            description=(
                "An explicitly started note-taking session for capturing and reviewing participant-authored notes. "
                "It does not provide general knowledge or inspect current-camera content."
            ),
        ),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        dialogue_budget_bytes=config.context_budget_bytes,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )

    async def publish_note_context(
        participant_id: str,
        update: ConversationContextUpdate,
    ) -> None:
        await runtime.publish(
            _NOTES_CONTEXT_TOPIC,
            update,
            participant_id=participant_id,
        )

    async def publish_monitor_window(
        participant_id: str,
        update: ConversationContextUpdate,
    ) -> None:
        await runtime.publish(
            MONITOR_WINDOW_TOPIC,
            update,
            participant_id=participant_id,
        )

    async def publish_observation_context(
        participant_id: str,
        update: ConversationContextUpdate,
    ) -> None:
        await runtime.publish(
            OBSERVATION_CONTEXT_TOPIC,
            update,
            participant_id=participant_id,
        )

    # Each destination owns its tools, admission policy, and sample state.
    observation = VisualObservationAgent(
        frames,
        vision,
        publish_observation_context,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    notes_agent = NoteTakingAgent(
        notes_conversation,
        decisions,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        context_budget_bytes=config.context_budget_bytes,
        publish_context=publish_note_context,
    )
    monitor = VisualMonitorAgent(
        frames,
        ImageQueryTool(images=images, vlm=vlm, system_prompt=_MONITOR_VLM_INSTRUCTIONS),
        decisions,
        publish_monitor_window,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        window_size=config.monitor_window_size,
        context_budget_bytes=config.context_budget_bytes,
    )

    async def initialize_participant(participant_id: str) -> None:
        await notes_agent.participant_joined(participant_id)
        await monitor.participant_joined(participant_id)
        await observation.participant_joined(participant_id)

    async def cleanup_participant(participant_id: str) -> None:
        # Routes used for turns are also cleaned by the router. These idempotent
        # hooks cover resources initialized at join but never selected for a turn.
        errors: list[Exception] = []
        for owner in (notes_agent, monitor, conversation, notes_conversation):
            try:
                await owner.participant_left(participant_id)
            except Exception as error:  # noqa: BLE001 -- retain independent cleanup failures.
                errors.append(error)
        try:
            frames.release(participant_id)
        except Exception as error:  # noqa: BLE001 -- retain the frame owner failure too.
            errors.append(error)
        if errors:
            raise ExceptionGroup(f"failed to clean up participant {participant_id!r}", errors)

    router = AgentRouter(
        {
            "conversation": conversation,
            "observation": observation,
            "monitor": monitor,
            "notes": notes_agent,
        },
        decisions,
        acknowledgment_llm=acknowledgment_llm,
        acknowledgment_delay_secs=config.acknowledgment_delay_secs,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        initialize_participant=initialize_participant,
        cleanup_participant=cleanup_participant,
        set_status=transport.endpoint.set_status,
    )
    # Only general conversation consumes the configured shared context sources.
    if config.summary_context_topics or config.incremental_context_topics:
        runtime.register("conversation-context", conversation.context_subscriber())
    runtime.register("conversation-example", router)
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
        await monitor.stop()
        if acknowledgment_llm is not None:
            await acknowledgment_llm.close()
        await llm.close()
    logger.info("conversation-example stopped")
