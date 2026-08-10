# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background speech transcript persistence and bounded periodic summaries."""

import json
import uuid
from pathlib import Path

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionRef, LLMRef
from xr_ai_nat.events import EventDispatcher, PeriodicEventSource

from ..agents.factory import build_agent
from ..agents.invoke import invoke_with_tool_retry
from ..agents.prompts import HUMAN
from ..runtime.events import emit
from ..runtime.scope import current_invocation, invocation_scope
from ..runtime.state import Session
from .controls import add_background_controls
from .events import (
    BACKGROUND_FACT,
    USER_OUTPUT,
    BackgroundFact,
    OutputDestination,
    UserOutput,
)
from .manager.runtime import ApplicationOwnership
from .manager.spec import ApplicationDescriptor
from .manager.types import RoutedFunction
from .output import UserOutputDelivery
from .transcript_store import TranscriptState, append_records, timestamp, transcript_path
from .transcript_summary import add_transcript_summary


class TranscriptApplication:
    def __init__(
        self,
        spec: ApplicationDescriptor,
        runtime: ApplicationOwnership,
        events: EventDispatcher,
        *,
        periodic: PeriodicEventSource | None = None,
    ) -> None:
        if spec.mode != "background":
            raise ValueError("transcript recorder must be a background application")
        self.spec = spec
        self.runtime = runtime
        self.events = events
        self.periodic = periodic
        self.output_dir = Path(spec.settings["output_dir"])
        self.summary_interval_s = float(spec.settings.get("summary_interval_s", 120))
        self.summary_prompt = str(spec.settings["summary_prompt"])
        if self.summary_interval_s <= 0:
            raise ValueError("summary_interval_s must be positive")
        self._states: dict[str, TranscriptState] = {}
        self._agent: Function | None = None

    @property
    def app_id(self) -> str:
        return self.spec.id

    async def build(
        self,
        builder: WorkflowBuilder,
        llm_ref: LLMRef,
    ) -> tuple[RoutedFunction, ...]:
        await add_transcript_summary(builder, self)
        self._agent = await build_agent(
            builder,
            name="background_transcript_summary",
            llm_ref=llm_ref,
            prompt=f"{self.summary_prompt}\n{HUMAN}",
            tools=(FunctionRef("transcript__commit_summary"),),
            return_direct=(FunctionRef("transcript__commit_summary"),),
        )
        return await add_background_controls(builder, self)

    async def start(self, session: Session, _instruction: str = "") -> str:
        if not self.runtime.start_background(session, self.app_id):
            return f"{self.spec.title} is already running."
        state = TranscriptState(
            path=transcript_path(self.output_dir, session.participant_id),
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
        emit(
            "transcript.started",
            participant_id=session.participant_id,
            path=str(state.path),
        )
        return f"{self.spec.title} started in the background."

    async def stop(self, session: Session) -> str:
        if not self.runtime.stop_background(session, self.app_id):
            return f"{self.spec.title} is not running."
        if self.periodic is not None:
            await self.periodic.stop(session.participant_id)
        state = self._states.get(session.participant_id)
        if state is not None:
            async with state.lock:
                state.active = False
                state.writes.append({"type": "session_end", "timestamp": timestamp()})
            await self._flush(state)
            if not state.summarizing:
                self._states.pop(session.participant_id, None)
        emit("transcript.stopped", participant_id=session.participant_id)
        return f"{self.spec.title} stopped."

    async def status(self, session: Session) -> str:
        state = "running" if self.runtime.is_background_active(session, self.app_id) else "stopped"
        return f"{self.spec.title} is {state}."

    async def on_transcription(self, session: Session, text: str, trace_id: str) -> None:
        state = self._states.get(session.participant_id)
        if state is None:
            return
        async with state.lock:
            if not state.active:
                return
            state.turns.append(text)
            state.writes.append(
                {
                    "type": "utterance",
                    "timestamp": timestamp(),
                    "trace_id": trace_id,
                    "text": text,
                }
            )
        emit(
            "transcript.utterance",
            participant_id=session.participant_id,
            trace_id=trace_id,
            characters=len(text),
        )
        await self._flush(state)

    async def tick(self, session: Session, trace_id: str | None = None) -> None:
        state = self._states.get(session.participant_id)
        if state is None:
            return
        async with state.lock:
            if not state.active or state.summarizing:
                return
            if not state.turns:
                return
            state.summarizing = True
            turns = tuple(state.turns)
        try:
            if self._agent is None:
                raise RuntimeError("transcript application has not been built")
            trace_id = trace_id or uuid.uuid4().hex[:12]
            with invocation_scope(session, trace_id):
                call = current_invocation()
                call.context["transcript.state"] = state
                call.context["transcript.turns"] = turns
                payload = json.dumps({"utterances": turns}, ensure_ascii=False, separators=(",", ":"))
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
        finally:
            async with state.lock:
                state.summarizing = False
                active = state.active
            if not active:
                self._states.pop(session.participant_id, None)

    async def commit_summary(
        self,
        session: Session,
        state: TranscriptState,
        turns: tuple[str, ...],
        summary: str,
    ) -> None:
        async with state.lock:
            if tuple(state.turns[: len(turns)]) != turns:
                raise ValueError("transcript summary no longer matches pending turns")
            del state.turns[: len(turns)]
            state.writes.append(
                {
                    "type": "summary",
                    "timestamp": timestamp(),
                    "turn_count": len(turns),
                    "text": summary.strip(),
                }
            )
            should_output = state.active
        await self._flush(state)
        emit(
            "transcript.summary",
            participant_id=session.participant_id,
            turn_count=len(turns),
            summary=summary.strip(),
        )
        if should_output:
            text = summary.strip()
            trace_id = current_invocation().trace_id
            await self.events.publish(
                BACKGROUND_FACT,
                participant_id=session.participant_id,
                producer=self.spec.title,
                payload=BackgroundFact(
                    topic="transcript.summary",
                    summary=text,
                    source_ref=str(state.path),
                ),
                correlation_id=trace_id,
            )
            await self.events.publish(
                USER_OUTPUT,
                participant_id=session.participant_id,
                producer=self.spec.title,
                payload=UserOutput(
                    text=text,
                    label=self.spec.title,
                    destinations=(OutputDestination.TEXT,),
                ),
                subscribers=(UserOutputDelivery.TEXT_SUBSCRIBER,),
                correlation_id=trace_id,
            )

    async def release(self, session: Session) -> None:
        if self.periodic is not None:
            await self.periodic.stop(session.participant_id)
        state = self._states.get(session.participant_id)
        if state is None:
            return
        async with state.lock:
            state.active = False
            state.writes.append({"type": "session_end", "timestamp": timestamp()})
        await self._flush(state)
        self._states.pop(session.participant_id, None)

    async def _flush(self, state: TranscriptState) -> None:
        async with state.lock:
            records = tuple(state.writes)
            state.writes.clear()
        if records:
            append_records(state.path, records)


__all__ = ["TranscriptApplication"]
