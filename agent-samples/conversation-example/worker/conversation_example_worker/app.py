# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the conversation example assistant from shared SDK primitives."""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from PIL import Image
from xr_ai_conversation import (
    Conversation,
    ConversationContextUpdate,
)
from xr_ai_routing import AgentRouter, RoutingAid
from xr_ai_logging import setup_logging
from xr_ai_models import (
    DecisionService,
    ModelsConfig,
    VLMService,
    load_models_config,
    make_decision,
    make_llm,
    make_tts,
    make_vlm,
)
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool, StreamingImageQueryTool
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from .agent import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ConversationExampleAgent,
)
from .config import WorkerConfig
from .conversation_agents import ContextualConversationAgent, NoteTakingAgent
from .monitoring import MONITOR_WINDOW_TOPIC, VisualMonitorAgent
from .visual_agents import VisualObservationAgent

_VLM_WARMUP_SIZE = (1280, 720)
_VLM_WARMUP_MAX_TOKENS = 4
_VLM_WARMUP_TIMEOUT_S = 120.0
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


def _vlm_warmup_jpeg() -> bytes:
    buffer = io.BytesIO()
    with Image.new("RGB", _VLM_WARMUP_SIZE, color=(128, 128, 128)) as image:
        image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


_VLM_WARMUP_IMAGE = _vlm_warmup_jpeg()


def _acknowledgment_models(models: ModelsConfig) -> ModelsConfig:
    """Clone the configured LLM role with reasoning disabled for brief acks."""

    spec = models.llm("llm")
    extras = dict(spec.adapter.default_extras)
    extras.pop("max_tokens", None)
    template = dict(extras.get("chat_template_kwargs", {}))
    template.pop("reasoning_budget", None)
    template.pop("thinking_budget", None)
    template["enable_thinking"] = False
    extras["chat_template_kwargs"] = template
    acknowledgment_spec = replace(spec, adapter=replace(spec.adapter, default_extras=extras))
    return ModelsConfig({**models.entries, "acknowledgment_llm": acknowledgment_spec})


async def _warm_vlm(vlm: VLMService) -> bool:
    """Confirm readiness by exercising the production multimodal stream."""

    try:
        started_at = time.monotonic()
        logger.info(
            "VLM warmup request started image={}x{} max_tokens={}",
            *_VLM_WARMUP_SIZE,
            _VLM_WARMUP_MAX_TOKENS,
        )
        first_token_at: float | None = None
        async for _ in vlm.stream_images(
            [_VLM_WARMUP_IMAGE],
            "What is the dominant color?",
            system_prompt="Answer with one word.",
            max_tokens=_VLM_WARMUP_MAX_TOKENS,
            timeout=_VLM_WARMUP_TIMEOUT_S,
        ):
            if first_token_at is None:
                first_token_at = time.monotonic()
                logger.info(
                    "VLM warmup first token latency_ms={:.1f}",
                    (first_token_at - started_at) * 1000,
                )
        logger.info(
            "VLM warmup completed total_ms={:.1f}",
            (time.monotonic() - started_at) * 1000,
        )
        return True
    except Exception:
        logger.exception("VLM warmup failed; readiness will retry")
        return False


@asynccontextmanager
async def _relay_event_log(log_file: Path) -> AsyncIterator[Path]:
    event_path = log_file.parent / "relay-events.jsonl"
    sink = event_path.open("w", encoding="utf-8")
    lock = Lock()
    subscriber = "conversation-example-compact-event-log"

    def write_event(event: nemo_relay.Event) -> None:
        if event.kind == "mark" and event.name == "llm.chunk":
            return
        with lock:
            sink.write(event.to_json())
            sink.write("\n")
            sink.flush()

    try:
        nemo_relay.subscribers.register(subscriber, write_event)
    except Exception:
        sink.close()
        raise
    try:
        yield event_path
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscriber)
        sink.close()


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
        probes={"llm": llm.health, "router": decisions.health, "vlm": lambda: _warm_vlm(vlm)},
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
    # General answers and focused notes use distinct instructions; AgentRouter supplies shared turn history.
    conversation = Conversation(
        llm,
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid(name="conversation", description=_CONVERSATION_SCOPE),
        summary_context_topics=config.summary_context_topics,
        incremental_context_topics=config.incremental_context_topics,
        context_budget_bytes=config.context_budget_bytes,
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
        context_budget_bytes=config.context_budget_bytes,
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

    # Each destination owns its tools, admission policy, and sample state.
    observation = VisualObservationAgent(
        frames,
        vision,
    )
    notes_agent = NoteTakingAgent(
        notes_conversation,
        decisions,
        context_budget_bytes=config.context_budget_bytes,
        publish_context=publish_note_context,
    )
    monitor = VisualMonitorAgent(
        frames,
        ImageQueryTool(images=images, vlm=vlm, system_prompt=_MONITOR_VLM_INSTRUCTIONS),
        decisions,
        publish_monitor_window,
        window_size=config.monitor_window_size,
        context_budget_bytes=config.context_budget_bytes,
    )
    conversation_agent = ContextualConversationAgent(
        conversation,
        static_description=_CONVERSATION_SCOPE,
        context_budget_bytes=config.context_budget_bytes,
        routing_context=monitor.routing_context,
        source_name="conversation",
    )
    router = AgentRouter(
        {
            "conversation": conversation_agent,
            "observation": observation,
            "monitor": monitor,
            "notes": notes_agent,
        },
        decisions,
        acknowledgment_llm=acknowledgment_llm,
        acknowledgment_delay_secs=config.acknowledgment_delay_secs,
    )
    # Only general conversation consumes the configured shared context sources.
    if config.summary_context_topics or config.incremental_context_topics:
        runtime.register("conversation-context", conversation.context_subscriber())
    # AgentRouter cleans routes used on turns. These direct callbacks
    # also clear participant-joined conversation and sample-owned state when a
    # participant leaves without routing through every destination.
    # This adapter translates voice queries into typed AgentRouter turns.
    conversation_example = runtime.register(
        "conversation-example",
        ConversationExampleAgent(
            transport.endpoint.set_status,
            router=router,
            conversations=(conversation, notes_conversation),
            frames=frames,
            participant_left_handlers=(notes_agent.participant_left, monitor.participant_left),
        ),
    )
    runtime.register("voice", voice)

    logger.info("Relay events → {}", log_file.parent / "relay-events.jsonl")
    logger.info("conversation-example starting")
    try:
        async with _relay_event_log(log_file):
            async with runtime:
                try:
                    await voice.run(runtime)
                finally:
                    await conversation_example.stop()
    finally:
        await conversation.close()
        await notes_conversation.close()
        await monitor.stop()
        if acknowledgment_llm is not None:
            await acknowledgment_llm.close()
        await llm.close()
    logger.info("conversation-example stopped")
