# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned background visual monitor with a bounded observation window."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import nemo_relay
from loguru import logger
from xr_ai_conversation import (
    Admission,
    ConversationContextUpdate,
    ConversationTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_hub import FrameUnavailable
from xr_ai_models import DecisionQuestion, DecisionService
from xr_ai_runtime import Topic
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

MONITOR_WINDOW_TOPIC = Topic("simple-vlm.monitor-window", ConversationContextUpdate)
_NEUTRAL_WINDOW = "Background visual monitoring has not been started."
_UNAVAILABLE = "The monitor couldn't read a usable frame. Monitoring has stopped."
_STARTED = "I'll start tracking the current view in the background."
_STOPPED = "Background visual monitoring has stopped."

WindowPublisher = Callable[[str, ConversationContextUpdate], Awaitable[None]]
Emit = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MonitorRoutingCue:
    """Small participant-scoped state for routing to active monitor coverage."""

    status: str
    task_scope: str | None
    observations_available: bool
    latest_recorded_at: str | None

    def render(self) -> str:
        if self.status == "active":
            if self.observations_available:
                return (
                    f"Active task, using the latest start/retarget instruction (retarget replaces the previous task): "
                    f"{self.task_scope}; timestamped observations are available through {self.latest_recorded_at}. "
                    "Use only if the current request falls within the resulting task."
                )
            return (
                "Active task, using the latest start/retarget instruction (retarget replaces the previous task): "
                f"{self.task_scope}; no completed observation is available yet."
            )
        if self.observations_available:
            return (
                f"Monitoring is {self.status}; historical observations through {self.latest_recorded_at} may be in "
                "conversation context, but they are not current-view evidence."
            )
        return f"Monitoring is {self.status}; no observation is available as current evidence."


class VisualMonitorDestination:
    """Capture text-only visual observations while explicitly enabled for a participant."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        image_query: ImageQueryTool,
        decisions: DecisionService,
        publish_window: WindowPublisher,
        *,
        window_size: int,
        context_budget_bytes: int,
    ) -> None:
        if window_size <= 0 or context_budget_bytes <= 0:
            raise ValueError("monitor window size and context budget must be positive")
        self._frames = frames
        self._image_query = image_query
        self._decisions = decisions
        self._publish_window = publish_window
        self._window_size = window_size
        self._context_budget_bytes = context_budget_bytes
        self._records: dict[str, deque[str]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._generations: dict[str, int] = {}
        self._instructions: dict[str, str] = {}
        self._failures: dict[str, str] = {}
        self._pending: dict[str, tuple[str, str]] = {}
        self._routing_cues: dict[str, MonitorRoutingCue] = {}

    def routing_aid(self, participant_id: str) -> RoutingAid:
        task = self._tasks.get(participant_id)
        active = task is not None and not task.done()
        description = (
            "Start, stop, or check the status of background visual monitoring. Content questions are handled by "
            "conversation or current-frame observation, not by this control destination."
        )
        eligibility = "active" if active else "available"
        return RoutingAid(name="monitor", description=description, eligibility=eligibility)

    async def admit(self, turn: ConversationTurn, *, focused: bool) -> Admission:
        self._pending.pop(turn.participant_id, None)
        task = self._tasks.get(turn.participant_id)
        active = task is not None and not task.done()
        if active:
            criteria = {
                "retarget": "The participant explicitly wants to change what the active background monitor tracks.",
                "stop": "The participant explicitly wants background visual monitoring to end.",
                "status": "The participant asks whether background visual monitoring is active or stopped.",
                "handoff": (
                    "The request asks about recorded observation content, the current camera view, or anything else "
                    "rather than monitor control or status."
                ),
            }
        else:
            criteria = {
                "start": "The participant explicitly asks to begin background visual monitoring.",
                "status": "The participant asks whether background visual monitoring is active or stopped.",
                "handoff": "The request is not a monitor start, stop, or status control.",
            }
        instructions = (
            "Classify only background-monitor controls: start, retarget, stop, status, or handoff. "
            "Do not answer questions about "
            "recorded observation content; conversation receives the timestamped window. Do not answer questions about "
            "what is visible now; direct observation obtains a fresh frame. Monitoring is background work and does not "
            "take conversation focus. Do not start it merely because a request mentions a visual subject."
        )
        routing_cue = self._routing_cues.get(turn.participant_id)
        state = {
            "request": turn.query,
            "completed_history": [{"user": item.user, "assistant": item.assistant} for item in turn.history],
            "monitor_active": active,
            "monitor_request": self._instructions.get(turn.participant_id, ""),
            "has_recorded_observations": routing_cue.observations_available if routing_cue else False,
            "latest_recorded_at": routing_cue.latest_recorded_at if routing_cue else None,
        }
        with nemo_relay.scope.scope(
            "simple-vlm.monitor-control",
            nemo_relay.ScopeType.Agent,
            input=state,
            metadata={"participant_id": turn.participant_id, "focused": focused},
        ) as scope:
            response = await self._decisions.decide(
                state,
                {"monitor_action": DecisionQuestion(instructions=instructions, criteria=criteria)},
            )
            action = response.answers["monitor_action"].choice
            nemo_relay.scope.event(
                "simple-vlm.monitor-control-result",
                handle=scope,
                data={"choice": action, "confidence": response.answers["monitor_action"].confidence},
                metadata={"participant_id": turn.participant_id},
            )
        if action == "handoff":
            return Admission("decline")
        if action not in criteria:
            raise ValueError(f"Clef returned invalid monitor action {action!r}")
        self._pending[turn.participant_id] = (turn.turn_id, action)
        return Admission("accept")

    async def answer(self, turn: ConversationTurn, *, emit: Emit) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)
        action = pending[1] if pending is not None and pending[0] == turn.turn_id else None
        if action == "stop":
            await self._stop_monitor(turn.participant_id)
            await emit("monitor", _STOPPED)
            return DestinationResult("answer", _STOPPED)
        if action in {"start", "retarget"}:
            if action == "retarget":
                await self._stop_monitor(turn.participant_id, publish=False)
            await emit("monitor", _STARTED)
            await self._start(turn)
            return DestinationResult("answer", _STARTED)
        if action == "status":
            text = self._status_text(turn.participant_id)
            await emit("monitor", text)
            return DestinationResult("answer", text)
        return DestinationResult("answer", "")

    async def _start(self, turn: ConversationTurn) -> None:
        participant_id = turn.participant_id
        generation = self._generations.get(participant_id, 0) + 1
        self._generations[participant_id] = generation
        self._records[participant_id] = deque(maxlen=self._window_size)
        instructions = _fit_utf8(turn.query, min(self._context_budget_bytes, 512))
        self._instructions[participant_id] = instructions
        self._failures.pop(participant_id, None)
        self._refresh_routing_cue(participant_id)
        try:
            # Replace the named summary before exposing the new task to sampling.
            await self._publish(participant_id)
        except asyncio.CancelledError:
            await self._rollback_start(participant_id, generation)
            raise
        except Exception:
            await self._rollback_start(participant_id, generation, failed=True)
            raise
        task = asyncio.create_task(
            self._observe(participant_id, generation, instructions),
            name=f"simple-vlm-monitor:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(pid, completed))

    async def _rollback_start(self, participant_id: str, generation: int, *, failed: bool = False) -> None:
        self._instructions.pop(participant_id, None)
        self._generations[participant_id] = generation + 1
        if failed:
            self._failures[participant_id] = "the monitor context could not be updated"
        self._refresh_routing_cue(participant_id)
        try:
            await self._publish(participant_id)
        except Exception:
            logger.exception("failed to clear incomplete monitor start participant={!r}", participant_id)

    async def _observe(self, participant_id: str, generation: int, instructions: str) -> None:
        try:
            while self._is_current(participant_id, generation, instructions):
                frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
                if not self._is_current(participant_id, generation, instructions):
                    return
                result = await self._image_query.execute(
                    ImageQueryRequest(
                        image=frame.image,
                        query=(
                            f"Monitoring request: {instructions}\nDescribe only relevant details visible in this frame."
                        ),
                    )
                )
                if not self._is_current(participant_id, generation, instructions):
                    return
                if not result.available or not result.text.strip():
                    raise RuntimeError(result.text or _UNAVAILABLE)
                timestamp = _timestamp(getattr(frame, "timestamp_us", None))
                record = f"{timestamp}: {result.text.strip()}"
                self._records.setdefault(participant_id, deque(maxlen=self._window_size)).append(record)
                self._refresh_routing_cue(participant_id)
                await self._publish(participant_id)
        except asyncio.CancelledError:
            raise
        except (FrameUnavailable, RuntimeError) as error:
            if not self._is_current(participant_id, generation, instructions):
                return
            self._failures[participant_id] = str(error) or _UNAVAILABLE
            self._instructions.pop(participant_id, None)
            self._generations[participant_id] = generation + 1
            self._refresh_routing_cue(participant_id)
            logger.warning("background visual monitor stopped participant={!r}: {}", participant_id, error)
            if participant_id in self._records:
                self._records[participant_id].append(f"Observation unavailable at {_timestamp(None)}.")
                await self._publish(participant_id)
        except Exception:
            if not self._is_current(participant_id, generation, instructions):
                return
            self._failures[participant_id] = "the observation request failed"
            self._instructions.pop(participant_id, None)
            self._generations[participant_id] = generation + 1
            self._refresh_routing_cue(participant_id)
            logger.exception("background visual monitor failed participant={!r}", participant_id)
            if participant_id in self._records:
                self._records[participant_id].append(f"Observation unavailable at {_timestamp(None)}.")
                await self._publish(participant_id)

    async def _publish(self, participant_id: str) -> None:
        active = participant_id in self._instructions
        status = "active" if active else "stopped"
        text = _render_window(
            self._records.get(participant_id, ()),
            self._context_budget_bytes,
            status=status,
            failure=bool(self._failures.get(participant_id)),
        )
        await self._publish_window(
            participant_id,
            ConversationContextUpdate(mode="summary", text=text or _NEUTRAL_WINDOW),
        )

    def routing_context(self, participant_id: str) -> str | None:
        cue = self._routing_cues.get(participant_id)
        return cue.render() if cue is not None else None

    def _refresh_routing_cue(self, participant_id: str) -> None:
        records = self._records.get(participant_id, ())
        active = participant_id in self._instructions
        if active:
            status = "active"
        elif self._failures.get(participant_id):
            status = "stopped after failure"
        elif records:
            status = "stopped"
        else:
            status = "not started"
        latest = next(
            (record for record in reversed(records) if not record.startswith("Observation unavailable")),
            None,
        )
        if latest:
            latest = latest.split(": ", maxsplit=1)[0]
        self._routing_cues[participant_id] = MonitorRoutingCue(
            status=status,
            task_scope=self._instructions.get(participant_id),
            observations_available=latest is not None,
            latest_recorded_at=latest,
        )

    def _is_current(self, participant_id: str, generation: int, instructions: str) -> bool:
        return (
            self._generations.get(participant_id) == generation
            and self._instructions.get(participant_id) == instructions
        )

    def _status_text(self, participant_id: str) -> str:
        cue = self._routing_cues.get(participant_id)
        if cue is None:
            return "Background visual monitoring has not been started."
        if cue.status == "active":
            if cue.latest_recorded_at:
                return f"Background monitoring is active; latest recorded observation: {cue.latest_recorded_at}."
            return "Background monitoring is active; no observation has completed yet."
        return cue.render()

    async def release(self, participant_id: str) -> None:
        """Keep independently requested background monitoring running on reroute."""

    async def _stop_monitor(self, participant_id: str, *, publish: bool = True) -> None:
        self._instructions.pop(participant_id, None)
        self._generations[participant_id] = self._generations.get(participant_id, 0) + 1
        self._failures.pop(participant_id, None)
        self._refresh_routing_cue(participant_id)
        await self._cancel(participant_id)
        if publish:
            await self._publish(participant_id)

    async def participant_left(self, participant_id: str) -> None:
        self._instructions.pop(participant_id, None)
        await self._cancel(participant_id)
        self._records.pop(participant_id, None)
        self._failures.pop(participant_id, None)
        self._routing_cues.pop(participant_id, None)
        self._generations.pop(participant_id, None)
        self._pending.pop(participant_id, None)

    async def stop(self) -> None:
        for participant_id in tuple(self._tasks):
            await self._stop_monitor(participant_id)

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("background visual monitor failed participant={!r}: {!r}", participant_id, error)


def _render_window(
    records: Iterable[str],
    budget_bytes: int,
    *,
    status: str,
    failure: bool,
) -> str:
    header = f"Background visual monitoring status: {status}."
    if failure:
        header += " The latest observation attempt failed; these records are historical."
    if len(header.encode("utf-8")) >= budget_bytes:
        return _fit_utf8(header, budget_bytes)
    selected: list[str] = []
    for record in reversed(tuple(records)):
        line = f"- {record}"
        candidate = "\n".join((header, *reversed([*selected, line])))
        if len(candidate.encode("utf-8")) > budget_bytes:
            if not selected:
                available = budget_bytes - len(header.encode("utf-8")) - 1
                selected.append(_fit_utf8(line, available))
            break
        selected.append(line)
    if not selected:
        return header
    rendered = "\n".join((header, *reversed(selected)))
    if len(rendered.encode("utf-8")) > budget_bytes:
        # The bounded latest record may be clipped only when it alone exceeds the
        # context allowance; older complete observations are dropped first.
        rendered = _fit_utf8(rendered, budget_bytes)
    return rendered


def _fit_utf8(text: str, budget_bytes: int) -> str:
    encoded = text.encode("utf-8")[:budget_bytes]
    return encoded.decode("utf-8", errors="ignore").strip()


def _timestamp(timestamp_us: int | None) -> str:
    if timestamp_us is None:
        return f"processing time {_utc_now()} (source capture time unknown)"
    value = datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).replace(second=0, microsecond=0)
    return value.strftime("%Y-%m-%d %H:%M UTC (frame source time)")


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


__all__ = ["MONITOR_WINDOW_TOPIC", "VisualMonitorDestination"]
