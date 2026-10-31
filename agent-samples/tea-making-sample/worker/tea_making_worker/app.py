# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire the sample's services, focused SOP agent, and SDK runtime.

``run_app`` is the composition entry point: create services, activate capture,
register background capabilities, then connect the SOP destination to AgentRouter
and voice. Model behavior comes from worker/workflow YAML, not this module.
Shutdown reverses ownership: stop routing before closing SOP context and producers,
then allow recording and service resources to finish.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_decision, make_llm, make_tts, make_vlm
from xr_ai_routing import AgentRouter
from xr_ai_runtime import AgentRuntime
from xr_ai_tools import ToolSet
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config
from xr_ai_web_events import WebEventsAgent

from ._capture import _record_routing, _RecordedDecisions, _RecordedLanguage, _RecordedVision, _SOPCapture
from ._cleanup import _CleanupAcks
from .agents._destinations import _destinations
from .agents.background_context import BackgroundContextAgent
from .agents.change_watch import ChangeWatchAgent
from .agents.file_output import FileOutputAgent
from .agents.images import ParticipantImageAgent
from .agents.sop.agent import SOPAgent
from .agents.transcript import TranscriptAgent
from .agents.video_log import VideoLogAgent
from .agents.web_events import TeaWebEventsAgent
from .config import WorkerConfig
from .events import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from .sop import load_workflow

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


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    """Compose and run the sample until voice stops, then drain owned resources.

    Args:
        config: Complete worker YAML configuration with resolved service/task paths.
        ready_file: Optional launcher signal written by voice after readiness checks.
    """

    # Create services before the runtime. Voice owns closing their underlying
    # connections; recording wrappers below delegate to these same services.
    setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)
    llm = make_llm(models, "llm")
    decisions = make_decision(models, "decision")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")
    rag = RAGTools(config.rag_endpoint)
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
        closeables=(llm, vlm, decisions, rag),
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
    workflow = load_workflow(config.workflow_config)
    capture = _SOPCapture(config.artifacts_dir, enabled=config.capture_enabled)
    await capture.start(config, workflow)
    llm = _RecordedLanguage(llm)
    decisions = _RecordedDecisions(decisions)
    vlm = _RecordedVision(vlm)
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
        files = runtime.register(
            "file-output",
            FileOutputAgent(
                capture.directory,
                history_size=config.background_history_size,
            ),
        )
        images = runtime.register(
            "images",
            ParticipantImageAgent(
                endpoint=transport.endpoint,
                frame_max_age_s=config.frame_max_age_s,
                frame_timeout_s=config.frame_timeout_s,
            ),
        )
        background_context = runtime.register(
            "background-context",
            BackgroundContextAgent(capacity=config.background_history_size),
        )
        change_watch = runtime.register(
            "change-watch",
            ChangeWatchAgent(
                images=images,
                vlm=vlm,
                llm=llm,
                decisions=decisions,
                decision=config.change_watch_decision,
                caption_prompt=config.change_watch_caption_prompt,
                event_prompt=config.change_watch_event_prompt,
                default_instruction=config.change_watch_default_instruction,
                interval_s=config.change_watch_interval_s,
            ),
        )
        transcript = runtime.register(
            "transcript",
            TranscriptAgent(
                llm=llm,
                summary_prompt=config.transcript_summary_prompt,
                summary_interval_s=config.transcript_summary_interval_s,
            ),
        )
        video_log = runtime.register(
            "video-log",
            VideoLogAgent(
                images=images,
                vlm=vlm,
                llm=llm,
                decisions=decisions,
                decision=config.video_delta_decision,
                caption_prompt=config.video_caption_prompt,
                delta_prompt=config.video_delta_prompt,
                interval_s=config.video_log_interval_s,
                history_size=5,
            ),
        )
        def application_tools(participant_id: str) -> ToolSet:
            """Collect the existing participant-scoped tools without owning their agents."""
            return ToolSet(
                tool
                for provider in (background_context, change_watch, transcript, video_log)
                for _, tool in provider.participant_tools(participant_id).items()
            )

        # Compose the focused SOP from existing services. Background capabilities
        # provide participant-scoped tools; they do not own workflow transitions.
        sop = runtime.register(
            "sop",
            SOPAgent(
                llm=llm,
                decisions=decisions,
                images=images,
                vlm=vlm,
                rag=rag,
                workflow=workflow,
                image_query=ImageQueryTool(images=images.images, vlm=vlm, system_prompt=config.current_view_prompt),
                context_budget_bytes=config.context_budget_bytes,
                compaction_input_budget_bytes=config.compaction_input_budget_bytes,
                application_tools=application_tools,
                prompt=config.foreground_prompt,
                current_view_prompt=config.current_view_prompt,
                vlm_timeout_s=config.vlm_timeout_s,
            ),
        )
        sop.bind_runtime(runtime)
        router = AgentRouter(
            _destinations(
                sop,
                policy=workflow.foreground,
                decisions=decisions,
                llm=llm,
                participant_tools=application_tools,
                respond=sop._respond,
                generate_answer=sop._generate_answer,
                current_frame=images.get_current_frame,
                vision=sop.tools[0],
                rag=rag,
                vlm_timeout_s=config.vlm_timeout_s,
            ),
            decisions,
            query_topic=USER_QUERY_TOPIC, output_topic=VOICE_OUTPUT_TOPIC,
            participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
            interrupted_topic=INTERRUPTED_TOPIC,
            initialize_participant=sop.participant_joined, cleanup_participant=sop.participant_left,
            set_status=partial(_record_routing, snapshot=sop._snapshot),
        )
        runtime.register("routing", router.runtime_subscriber())
        runtime.register("sop-cleanup", _CleanupAcks(router))
        runtime.register("voice", voice)

        logger.info("SOP run bundle → {}", capture.directory)
        logger.info("tea-making starting with Omni for language and vision")
        async with _relay_event_log(capture.directory):
            async with web_events:
                logger.info("live events → {}", web_events.url)
                async with runtime:
                    change_watch.bind_runtime(runtime)
                    transcript.bind_runtime(runtime)
                    video_log.bind_runtime(runtime)
                    try:
                        await voice.run(runtime)
                    finally:
                        await router.stop()
                        await sop.stop()
                        await change_watch.stop()
                        await transcript.stop()
                        await video_log.stop()
                        await images.stop()
                        await files.stop()

    logger.info("tea-making stopped")


__all__ = ["run_app"]
