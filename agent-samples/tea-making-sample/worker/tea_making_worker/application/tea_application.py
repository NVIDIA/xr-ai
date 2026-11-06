# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire the sample's services, focused SOP agent, and SDK runtime.

``run_app`` is the composition entry point: create services, activate capture,
register camera and SOP event infrastructure, then connect the SOP destination to AgentRouter
and voice. Model behavior comes from application YAML, not this module.
Shutdown reverses ownership: stop routing before closing SOP context and producers,
then allow recording and service resources to finish.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path

import nemo_relay
import yaml
from loguru import logger
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_logging import setup_logging
from pydantic import BaseModel
from xr_ai_models import ModelsConfig, load_models_config, make_decision, make_llm, make_tts, make_vlm
from xr_ai_routing import AgentRouter, RoutingAid
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config
from xr_ai_web_events import WebEventsAgent

from ..components.sop_run_recording import _RecordedDecisions, _RecordedLanguage, _RecordedVision, _SOPCapture
from ..components.vision_rag_retrieval_flow import _without_schema_defaults
from .tea_data_models import TeaIdentity, _TemperatureObservation
from .application_config import WorkerConfig
from .tea_event_viewer import TeaWebEventsAgent
from .tea_events import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from .tea_sop_agent import SOP_CONTEXT_SOURCE, SOPAgent
from .tea_interaction import _TeaInteraction

_CLIENT_TEXT_TOPIC = "agent.response"


@asynccontextmanager
async def _relay_event_log(output_dir: Path) -> AsyncIterator[Path]:
    """Write relay events in batches and drain the writer before closing the log.

    The relay subscriber may run outside the event-loop thread, so it enqueues through
    ``call_soon_threadsafe``. The sentinel follows subscriber flush/deregistration.
    """
    path = output_dir / "relay-events.jsonl"
    await asyncio.to_thread(path.write_text, "", encoding="utf-8")
    pending_events: asyncio.Queue[str | None] = asyncio.Queue()
    event_loop = asyncio.get_running_loop()
    subscriber_name = "tea-making-event-log"

    def write_event(event: nemo_relay.Event) -> None:
        if event.kind == "mark" and event.name == "llm.chunk":
            return
        event_loop.call_soon_threadsafe(pending_events.put_nowait, f"{event.to_json()}\n")

    async def write_events() -> None:
        while True:
            line = await pending_events.get()
            if line is None:
                return
            lines = [line]
            while True:
                try:
                    queued_line = pending_events.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if queued_line is None:
                    await asyncio.to_thread(_append_relay_events, path, tuple(lines))
                    return
                lines.append(queued_line)
            await asyncio.to_thread(_append_relay_events, path, tuple(lines))

    writer_task = asyncio.create_task(write_events(), name="tea-making-relay-event-log")

    try:
        nemo_relay.subscribers.register(subscriber_name, write_event)
    except Exception:
        pending_events.put_nowait(None)
        await writer_task
        raise
    try:
        yield path
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscriber_name)
        await asyncio.sleep(0)
        pending_events.put_nowait(None)
        await writer_task


def _append_relay_events(path: Path, lines: tuple[str, ...]) -> None:
    """Append one ordered batch on the file-writing worker thread."""
    with path.open("a", encoding="utf-8") as sink:
        sink.writelines(lines)


def _structured_reader_models_config(
    models: ModelsConfig, name: str, observation_model: type[BaseModel]
) -> ModelsConfig:
    """Derive an application reader role without changing shared model settings."""
    # Each structured reader uses the same endpoint/weights and inference
    # settings; prose observation requests keep their existing client.
    identification_spec = models.vlm("vlm")
    identification_spec = replace(
        identification_spec,
        adapter=replace(identification_spec.adapter, default_extras={
            **identification_spec.adapter.default_extras,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": observation_model.__name__.lstrip("_"), "schema": _without_schema_defaults(observation_model.model_json_schema()), "strict": True,
            }},
        }),
    )
    return replace(models, entries={**models.entries, name: identification_spec})


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    """Compose and run the sample until voice stops, then drain owned resources.

    Args:
        config: Complete worker YAML configuration with resolved service/task paths.
        ready_file: Optional launcher signal written by voice after readiness checks.
    """

    # Shared services outlive all SOP and Conversation work. Voice owns only
    # its speech clients; application cleanup closes borrowed clients after drain.
    setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)
    shared_clients = []
    shared_work_drained = True
    try:
        llm = make_llm(models, "llm")
        shared_clients.append(llm)
        decisions = make_decision(models, "decision")
        shared_clients.append(decisions)
        vlm = make_vlm(models, "vlm")
        shared_clients.append(vlm)
        identification_models = _structured_reader_models_config(models, "identification", TeaIdentity)
        identification_vlm = make_vlm(identification_models, "identification")
        shared_clients.append(identification_vlm)
        temperature_models = _structured_reader_models_config(models, "temperature", _TemperatureObservation)
        temperature_vlm = make_vlm(temperature_models, "temperature")
        shared_clients.append(temperature_vlm)
        tts = make_tts(models, "tts")
        rag = RAGTools(config.rag_endpoint)
        shared_clients.append(rag)
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
            probes={"rag": rag.health, "clef": decisions.health},
            ready_file=ready_file,
            text_topic=_CLIENT_TEXT_TOPIC,
            idle_timeout_secs=config.idle_timeout_secs,
            transport=transport,
            participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
            participant_left_topic=PARTICIPANT_LEFT_TOPIC,
            interrupted_topic=INTERRUPTED_TOPIC,
            interrupt_on_supersede=True,
        )

        # Snapshot the complete effective configuration before any agent produces
        # evidence. All later model calls inherit this capture scope.
        capture = _SOPCapture(config.artifacts_dir, enabled=config.capture_enabled)
        await capture.start(
            configuration_snapshots={
                "worker": config,
                "models": json.loads(config.models_config.read_text()),
                "identification_models": identification_models,
                "temperature_models": temperature_models,
                "voice_gate": yaml.safe_load(config.voice_gate_yaml.read_text()),
            },
            source_directory=Path(__file__).resolve().parents[1],
            workflow_name="tea making",
            steps=("identify", "fill_water", "heat_water", "start_steeping", "steep_timer"),
        )
        llm = _RecordedLanguage(llm)
        conversation_llm = llm
        decisions = _RecordedDecisions(decisions)
        vlm = _RecordedVision(vlm)
        identification_vlm = _RecordedVision(identification_vlm)
        temperature_vlm = _RecordedVision(temperature_vlm)
        async with capture.activate():
            runtime = AgentRuntime()
            web_events = runtime.register(
                "web-events",
                WebEventsAgent(
                    host=config.web_events_host,
                    port=config.web_events_port,
                    max_events=config.web_events_max_events,
                    title="Tea-making agent events",
                ),
            )
            runtime.register("web-event-adapter", TeaWebEventsAgent())
            images = ImageRegistry()
            frames = CurrentFrameTool(
                endpoint=transport.endpoint,
                images=images,
                frame_max_age_s=config.frame_max_age_s,
                frame_timeout_s=config.frame_timeout_s,
            )
            # Conversation owns dialogue. Only the runner supplies the replaceable
            # SOP source; independent flows never connect to this manager.
            context = ContextManager(
                conversation_llm,
                summary_context_topics=(SOP_CONTEXT_SOURCE,),
                dialogue_budget_bytes=config.context_budget_bytes,
                topic_budget_bytes=config.sop_context_budget_bytes,
                compaction_input_budget_bytes=config.compaction_input_budget_bytes,
            )
            runtime.register("sop-conversation-context", context.subscriber())
            conversation = Conversation(
                conversation_llm,
                context=context,
                system_prompt=config.conversation_system_prompt,
                instructions=config.sop.conversation_instructions,
                routing=RoutingAid("conversation", config.sop.conversation_description),
                runtime=runtime,
                output_topic=VOICE_OUTPUT_TOPIC,
            )
            sop = runtime.register(
                "sop",
                SOPAgent(
                    config=config.sop,
                    llm=llm,
                    decisions=decisions,
                    frames=frames,
                    temperature_image_query=ImageQueryTool(
                        images=images, vlm=temperature_vlm, system_prompt=config.current_view_prompt,
                    ),
                    identification_image_query=ImageQueryTool(
                        images=images, vlm=identification_vlm, system_prompt=config.current_view_prompt,
                    ),
                    image_query=ImageQueryTool(
                        images=images,
                        vlm=vlm,
                        system_prompt=config.current_view_prompt,
                    ),
                    rag=rag,
                    context=context,
                    runtime=runtime,
                ),
            )

            async def cleanup_participant(participant_id: str) -> None:
                """Drain SOP work before releasing SDK frames; the router serializes rejoins."""
                await sop.participant_left(participant_id)
                frames.release(participant_id)

            router = AgentRouter(
                {"tea": _TeaInteraction(sop, conversation, decisions)},
                decisions,
                default_destination="tea",
                query_topic=USER_QUERY_TOPIC,
                output_topic=VOICE_OUTPUT_TOPIC,
                participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
                participant_left_topic=PARTICIPANT_LEFT_TOPIC,
                interrupted_topic=INTERRUPTED_TOPIC,
                initialize_participant=sop.participant_joined,
                cleanup_participant=cleanup_participant,
                set_status=transport.endpoint.set_status,
            )
            runtime.register("routing", router.runtime_subscriber())
            runtime.register("voice", voice)

            logger.info("SOP run bundle → {}", capture.directory)
            logger.info("tea-making starting with Omni for language and vision")
            async with _relay_event_log(capture.directory):
                async with web_events:
                    logger.info("live events → {}", web_events.url)
                    async with runtime:
                        try:
                            shared_work_drained = False
                            await voice.run(runtime)
                        finally:
                            await router.stop()
                            await sop.stop()
                            await conversation.close()
                            await context.close()

                            images.clear()
                            shared_work_drained = True
    finally:
        # Failed drain retains its clients rather than destroying resources
        # still owned by cancellation-resistant workflow work.
        if shared_work_drained:
            async with AsyncExitStack() as service_cleanup:
                registered_clients = set()
                for client in shared_clients:
                    if id(client) not in registered_clients:
                        registered_clients.add(id(client))
                        service_cleanup.push_async_callback(client.close)

    logger.info("tea-making stopped")


__all__ = ["run_app"]
