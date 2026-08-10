# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background broad-scene captioning with rolling delta persistence."""

import json
import uuid
from collections import deque
from pathlib import Path

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionRef, LLMRef
from xr_ai_nat.events import EventDispatcher, PeriodicEventSource
from xr_ai_nat.functions.vision import LiveVisionResult

from ..agents.factory import build_agent
from ..agents.invoke import invoke_with_tool_retry
from ..agents.prompts import HUMAN
from ..runtime.events import emit
from ..runtime.scope import current_invocation, invocation_scope
from ..runtime.state import Session
from .controls import add_background_controls
from .events import BACKGROUND_FACT, BackgroundFact
from .jsonl import append_records, session_path, timestamp
from .manager.runtime import ApplicationOwnership
from .manager.spec import ApplicationDescriptor
from .manager.types import RoutedFunction
from .video_log_delta import add_video_log_commit
from .video_log_store import VideoLogState


class VideoLogApplication:
    def __init__(
        self,
        spec: ApplicationDescriptor,
        runtime: ApplicationOwnership,
        events: EventDispatcher,
        *,
        periodic: PeriodicEventSource | None = None,
    ) -> None:
        if spec.mode != "background":
            raise ValueError("video logger must be a background application")
        self.spec = spec
        self.runtime = runtime
        self.events = events
        self.periodic = periodic
        self.output_dir = Path(spec.settings["output_dir"])
        self.interval_s = float(spec.settings.get("interval_s", 2))
        self.history_size = int(spec.settings.get("history_size", 5))
        self.caption_prompt = str(spec.settings["caption_prompt"])
        self.delta_prompt = str(spec.settings["delta_prompt"])
        if self.interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if self.history_size < 2:
            raise ValueError("history_size must be at least two")
        self._states: dict[str, VideoLogState] = {}
        self._view: Function | None = None
        self._agent: Function | None = None

    @property
    def app_id(self) -> str:
        return self.spec.id

    async def build(
        self,
        builder: WorkflowBuilder,
        llm_ref: LLMRef,
        current_view: Function,
    ) -> tuple[RoutedFunction, ...]:
        self._view = current_view
        await add_video_log_commit(builder, self)
        self._agent = await build_agent(
            builder,
            name="background_video_log_delta",
            llm_ref=llm_ref,
            prompt=f"{self.delta_prompt}\n{HUMAN}",
            tools=(FunctionRef("video_log__commit"),),
            return_direct=(FunctionRef("video_log__commit"),),
        )
        return await add_background_controls(builder, self)

    async def start(self, session: Session, _instruction: str = "") -> str:
        if not self.runtime.start_background(session, self.app_id):
            return f"{self.spec.title} is already running."
        state = VideoLogState(
            path=session_path(self.output_dir, session.participant_id),
            captions=deque(maxlen=self.history_size),
        )
        state.writes.append(
            {
                "type": "session",
                "timestamp": timestamp(),
                "participant_id": session.participant_id,
            }
        )
        self._states[session.participant_id] = state
        await self._flush(state)
        if self.periodic is not None:
            self.periodic.start(session.participant_id)
        emit("video_log.started", participant_id=session.participant_id, path=str(state.path))
        return f"{self.spec.title} started in the background."

    async def stop(self, session: Session) -> str:
        if not self.runtime.stop_background(session, self.app_id):
            return f"{self.spec.title} is not running."
        if self.periodic is not None:
            await self.periodic.stop(session.participant_id)
        state = self._states.pop(session.participant_id, None)
        if state is not None:
            async with state.lock:
                state.active = False
                state.writes.append({"type": "session_end", "timestamp": timestamp()})
            await self._flush(state)
        emit("video_log.stopped", participant_id=session.participant_id)
        return f"{self.spec.title} stopped."

    async def status(self, session: Session) -> str:
        state = self._states.get(session.participant_id)
        if state is None or not self.runtime.is_background_active(session, self.app_id):
            return f"{self.spec.title} is stopped."
        return f"{self.spec.title} is running."

    async def on_transcription(self, session: Session, text: str, trace_id: str) -> None:
        return None

    async def tick(self, session: Session, trace_id: str | None = None) -> None:
        state = self._states.get(session.participant_id)
        if state is None or self._view is None or self._agent is None:
            return
        if state.lock.locked():
            return
        async with state.lock:
            if not state.active:
                return
            trace_id = trace_id or uuid.uuid4().hex[:12]
            with invocation_scope(session, trace_id):
                result = await self._view.ainvoke(
                    {"question": self.caption_prompt},
                    to_type=LiveVisionResult,
                )
                caption = result.answer.strip()
                emit(
                    "video_log.caption",
                    participant_id=session.participant_id,
                    trace_id=trace_id,
                    caption=caption,
                )
                if caption.startswith("Unable to inspect"):
                    return
                current_invocation().context.update({"video_log.state": state, "video_log.caption": caption})
                previous = list(state.captions)[-(self.history_size - 1) :]
                payload = json.dumps(
                    {"previous": previous, "current": caption},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                emit(
                    "agent.background.request",
                    participant_id=session.participant_id,
                    application=self.app_id,
                    trace_id=trace_id,
                    input=payload,
                )

                def retry(feedback: str) -> None:
                    emit(
                        "agent.background.retry",
                        participant_id=session.participant_id,
                        application=self.app_id,
                        trace_id=trace_id,
                        feedback=feedback,
                    )

                output = await invoke_with_tool_retry(
                    self._agent,
                    payload,
                    retry=retry,
                    skip_repeated_invalid=True,
                )
                emit(
                    "agent.background.response" if output else "agent.background.skipped",
                    participant_id=session.participant_id,
                    application=self.app_id,
                    trace_id=trace_id,
                    output=output,
                )
        await self._flush(state)

    async def commit(
        self,
        session: Session,
        state: VideoLogState,
        caption: str,
        trace_id: str,
        delta: str,
    ) -> None:
        state.captions.append(caption)
        state.writes.append(
            {
                "type": "observation",
                "timestamp": timestamp(),
                "trace_id": trace_id,
                "caption": caption,
                "delta": delta,
            }
        )
        emit(
            "video_log.delta",
            participant_id=session.participant_id,
            trace_id=trace_id,
            delta=delta,
            history=list(state.captions),
        )
        if delta.rstrip(". ").casefold() != "no meaningful visual change":
            await self.events.publish(
                BACKGROUND_FACT,
                participant_id=session.participant_id,
                producer=self.spec.title,
                payload=BackgroundFact(
                    topic="video_log.delta",
                    summary=delta,
                    source_ref=str(state.path),
                ),
                correlation_id=trace_id,
            )

    async def release(self, session: Session) -> None:
        if self.periodic is not None:
            await self.periodic.stop(session.participant_id)
        state = self._states.pop(session.participant_id, None)
        if state is None:
            return
        async with state.lock:
            state.active = False
            state.writes.append({"type": "session_end", "timestamp": timestamp()})
        await self._flush(state)

    async def _flush(self, state: VideoLogState) -> None:
        async with state.lock:
            records = tuple(state.writes)
            state.writes.clear()
        if records:
            append_records(state.path, records)


__all__ = ["VideoLogApplication"]
