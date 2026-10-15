# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the simple VLM assistant from shared SDK primitives."""

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
    ConversationFrontEnd,
    RoutingAid,
)
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

from .admission import ClefAdmissionDestination
from .agent import (
    _CONVERSATION_INSTRUCTIONS,
    _ENVIRONMENT_INSTRUCTIONS,
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    SimpleVlmAgent,
)
from .config import WorkerConfig
from .visual_destinations import EnvironmentReasoningDestination, VisualObservationDestination

_VLM_WARMUP_SIZE = (1280, 720)
_VLM_WARMUP_MAX_TOKENS = 4
_VLM_WARMUP_TIMEOUT_S = 120.0
_ENVIRONMENT_EVIDENCE_TOPIC = Topic(
    "simple-vlm.environment-evidence",
    ConversationContextUpdate,
)


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
    subscriber = "simple-vlm-compact-event-log"

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
    """Run the worker until the voice session shuts down."""

    log_file = setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)

    llm = make_llm(models, "llm")
    acknowledgment_llm = (
        make_llm(_acknowledgment_models(models), "acknowledgment_llm")
        if config.acknowledgment_delay_secs is not None
        else None
    )
    router: DecisionService = make_decision(models, "router")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")

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
        probes={"llm": llm.health, "router": router.health, "vlm": lambda: _warm_vlm(vlm)},
        ready_file=ready_file,
        closeables=(vlm, router),
        text_topic="vlm.response",
        idle_timeout_secs=config.idle_timeout_secs,
        transport=transport,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )

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
    conversation = Conversation(
        llm,
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid(
            name="conversation",
            description="General conversation, explanations, calculations, and recall from completed exchanges.",
        ),
        context_budget_bytes=config.context_budget_bytes,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )
    environment_conversation = Conversation(
        llm,
        instructions=_ENVIRONMENT_INSTRUCTIONS,
        routing=RoutingAid(
            name="environment",
            description=("Explain, compare, recommend, or plan using current-scene evidence and conversation context."),
        ),
        context_budget_bytes=config.context_budget_bytes,
        compaction_input_budget_bytes=config.compaction_input_budget_bytes,
    )

    async def publish_environment_evidence(
        participant_id: str,
        update: ConversationContextUpdate,
    ) -> None:
        await runtime.publish(
            _ENVIRONMENT_EVIDENCE_TOPIC,
            update,
            participant_id=participant_id,
            source="simple-vlm.environment",
        )

    observation = VisualObservationDestination(
        frames,
        vision,
        context_budget_bytes=config.context_budget_bytes,
    )
    environment = EnvironmentReasoningDestination(
        llm,
        frames,
        ImageQueryTool(images=images, vlm=vlm, system_prompt=config.system_prompt),
        environment_conversation,
        publish_environment_evidence,
        context_budget_bytes=config.context_budget_bytes,
    )
    front_end = ConversationFrontEnd(
        {
            "conversation": ClefAdmissionDestination(conversation, router),
            "observation": ClefAdmissionDestination(observation, router),
            "environment": ClefAdmissionDestination(environment, router),
        },
        router,
        acknowledgment_llm=acknowledgment_llm,
        acknowledgment_delay_secs=config.acknowledgment_delay_secs,
    )
    configured_topics = tuple(Topic(name, ConversationContextUpdate) for name in config.context_topics)
    if configured_topics:
        runtime.register("conversation-context", conversation.context_subscriber(configured_topics))
    environment_topics = tuple(
        {topic.name: topic for topic in (*configured_topics, _ENVIRONMENT_EVIDENCE_TOPIC)}.values()
    )
    runtime.register(
        "environment-context",
        environment_conversation.context_subscriber(environment_topics),
    )
    simple_vlm = runtime.register(
        "simple-vlm",
        SimpleVlmAgent(
            transport.endpoint.set_status,
            front_end=front_end,
            conversations=(conversation, environment_conversation),
            frames=frames,
        ),
    )
    runtime.register("voice", voice)

    logger.info("Relay events → {}", log_file.parent / "relay-events.jsonl")
    logger.info("simple-vlm-example starting")
    try:
        async with _relay_event_log(log_file):
            async with runtime:
                try:
                    await voice.run(runtime)
                finally:
                    await simple_vlm.stop()
    finally:
        await conversation.close()
        await environment_conversation.close()
        if acknowledgment_llm is not None:
            await acknowledgment_llm.close()
        await llm.close()
    logger.info("simple-vlm-example stopped")
