# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background visual captions with prompt-driven change detection."""

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionRef, LLMRef
from xr_ai_nat.functions.vision import LiveVisionResult

from ..agents.factory import build_agent
from ..agents.invoke import invoke_with_tool_retry
from ..agents.prompts import HUMAN
from ..desktop.runtime import DesktopRuntime
from ..desktop.spec import ApplicationSpec
from ..desktop.types import RoutedFunction
from ..runtime.events import emit
from ..runtime.scope import current_invocation, invocation_scope
from ..runtime.state import Session
from .background import TextOutput
from .change_events import ChangeCommitRequest, add_change_commit
from .controls import add_background_controls
from .jsonl import append_records, session_path, timestamp


@dataclass(slots=True)
class _WatchState:
    path: Path
    captions: deque[str]
    instruction: str
    next_tick: float = 0.0
    writes: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChangeWatchApplication:
    def __init__(self, spec: ApplicationSpec, runtime: DesktopRuntime, output: TextOutput) -> None:
        if spec.mode != "background":
            raise ValueError("change watch must be a background application")
        self.spec = spec
        self.runtime = runtime
        self.output = output
        self.output_dir = Path(spec.settings["output_dir"])
        self.interval_s = float(spec.settings.get("interval_s", 2))
        self.history_size = int(spec.settings.get("history_size", 2))
        self.default_instruction = str(spec.settings["default_instruction"])
        self.caption_prompt = str(spec.settings["caption_prompt"])
        self.event_prompt = str(spec.settings["event_prompt"])
        self._states: dict[str, _WatchState] = {}
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
        await add_change_commit(builder, self)
        self._agent = await build_agent(
            builder,
            name="background_change_watch",
            llm_ref=llm_ref,
            prompt=f"{self.event_prompt}\n{HUMAN}",
            tools=(FunctionRef("change_watch__commit"),),
            return_direct=(FunctionRef("change_watch__commit"),),
        )
        return await add_background_controls(builder, self)

    async def start(self, session: Session, instruction: str = "") -> str:
        if not self.runtime.start_background(session, self.app_id):
            state = self._states[session.participant_id]
            return f"{self.spec.title} is already running. Monitoring: {state.instruction}."
        focus = instruction.rstrip(".!? ") or self.default_instruction.rstrip(".!? ")
        state = _WatchState(
            path=session_path(self.output_dir, session.participant_id),
            captions=deque(maxlen=self.history_size),
            instruction=focus,
        )
        state.writes.append(
            {
                "type": "session",
                "timestamp": timestamp(),
                "participant_id": session.participant_id,
                "watch_for": focus,
            }
        )
        self._states[session.participant_id] = state
        await self._flush(state)
        emit(
            "change_watch.started",
            participant_id=session.participant_id,
            instruction=focus,
            path=str(state.path),
        )
        return f"{self.spec.title} started in the background. Monitoring: {focus}."

    async def stop(self, session: Session) -> str:
        if not self.runtime.stop_background(session, self.app_id):
            return f"{self.spec.title} is not running."
        state = self._states.pop(session.participant_id, None)
        if state is not None:
            async with state.lock:
                state.active = False
                state.writes.append({"type": "session_end", "timestamp": timestamp()})
            await self._flush(state)
        emit("change_watch.stopped", participant_id=session.participant_id)
        return f"{self.spec.title} stopped."

    async def status(self, session: Session) -> str:
        state = self._states.get(session.participant_id)
        if state is None or not self.runtime.is_background_active(session, self.app_id):
            return f"{self.spec.title} is stopped."
        return f"{self.spec.title} is running. Monitoring: {state.instruction}."

    async def on_transcription(self, session: Session, text: str, trace_id: str) -> None:
        return None

    async def tick(self, session: Session) -> None:
        state = self._states.get(session.participant_id)
        if state is None or self._view is None or self._agent is None:
            return
        now = time.monotonic()
        if state.next_tick > now or state.lock.locked():
            return
        async with state.lock:
            if not state.active:
                return
            state.next_tick = now + self.interval_s
            trace_id = uuid.uuid4().hex[:12]
            with invocation_scope(session, trace_id):
                result = await self._view.ainvoke(
                    {"question": f"{self.caption_prompt}\nFocus: {state.instruction}"},
                    to_type=LiveVisionResult,
                )
                caption = result.answer.strip()
                emit(
                    "change_watch.caption",
                    participant_id=session.participant_id,
                    trace_id=trace_id,
                    caption=caption,
                )
                if caption.startswith("Unable to inspect"):
                    return
                if not state.captions:
                    state.captions.append(caption)
                    state.writes.append(
                        {
                            "type": "baseline",
                            "timestamp": timestamp(),
                            "trace_id": trace_id,
                            "watch_for": state.instruction,
                            "caption": caption,
                        }
                    )
                    emit(
                        "change_watch.baseline",
                        participant_id=session.participant_id,
                        trace_id=trace_id,
                        caption=caption,
                    )
                else:
                    call = current_invocation()
                    call.context["change_watch.caption"] = caption
                    payload = json.dumps(
                        {
                            "watch_for": state.instruction,
                            "previous": list(state.captions),
                            "current": caption,
                        },
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

    async def commit(self, session: Session, caption: str, request: ChangeCommitRequest) -> None:
        state = self._states[session.participant_id]
        state.captions.append(caption)
        summary = request.summary.strip() if request.important else ""
        trace_id = current_invocation().trace_id
        state.writes.append(
            {
                "type": "observation",
                "timestamp": timestamp(),
                "trace_id": trace_id,
                "watch_for": state.instruction,
                "caption": caption,
                "important": request.important,
                "summary": summary,
            }
        )
        emit(
            "change_watch.event",
            participant_id=session.participant_id,
            important=request.important,
            summary=summary,
            history=list(state.captions),
        )
        if summary:
            await self.output(session.participant_id, self.spec.title, summary)

    async def release(self, session: Session) -> None:
        state = self._states.pop(session.participant_id, None)
        if state is None:
            return
        async with state.lock:
            state.active = False
            state.writes.append({"type": "session_end", "timestamp": timestamp()})
        await self._flush(state)

    async def _flush(self, state: _WatchState) -> None:
        async with state.lock:
            records = tuple(state.writes)
            state.writes.clear()
        if records:
            append_records(state.path, records)


__all__ = ["ChangeWatchApplication"]
