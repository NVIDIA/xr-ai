# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own participant-scoped, one-shot background visual alerts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_conversation import ContextManager
from xr_ai_interaction import ResponseOutput, ResponseOutputs
from xr_ai_models import DecisionQuestion, DecisionService, ImageInput, LLMService, VLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutableAgent, RoutingAid
from xr_ai_runtime import AgentRuntime, RuntimeClosedError, Topic
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry

from ._alert_checks import AlertChecks, Trigger
from ._alert_compile import CompiledAlert, compile_alert
from ._decisions import decide
from ._tasks import wait_despite_cancellation

_ROUTING = (
    "Set a one-shot background visual alert ONLY when explicitly asked to notify the user when a specific "
    "visible condition occurs, including a condition they gave earlier or noted; cancel an existing alert on an "
    "explicit request. Visual questions, observations, recall, and fragments are not alert controls."
)
_PROMPTS = Path(__file__).parent / "prompts"
_CONTROL = DecisionQuestion(
    instructions=(_PROMPTS / "alert_control.txt").read_text(encoding="utf-8").strip(),
    criteria={
        "start": (
            "An explicit request to be alerted later when a specific visually observable condition is "
            "seen, stated directly or clearly referring to one the participant gave earlier."
        ),
        "cancel": "An explicit request to cancel or stop a visual alert.",
        "decline": (
            "Anything else, including current visual questions, observations, ambiguous fragments, "
            "capability questions, or an alert with neither a stated nor a referenced condition."
        ),
    },
)
_DESCRIPTION = (_PROMPTS / "alert_description.txt").read_text(encoding="utf-8").strip()
_ALERT = "Alert"
_CANCELLED = "The visual alert is cancelled."
_UNCLEAR = "I couldn't tell what to watch for. Please say what should trigger the alert."
_STOPPED = "I couldn't check the visual alert. It has stopped."
_NO_ANSWER = DestinationResult("answer", "")


@dataclass
class _Job:
    alert: CompiledAlert
    task: asyncio.Task[None] | None = None


class VisualAlertAgent(RoutableAgent):
    """Watch one condition per participant; announce and describe its triggering frame.

    A request is first compiled with the participant's recent exchanges and saved context into a
    standalone condition, so an alert can refer to something said or noted earlier. The watch then
    checks paced frames: each passes a cheap Clef relevance screen before the costly confirmation,
    which races a direct Clef image decision against a Cosmos observation judged by Clef.

    Only an explicit new alert replaces a job. Rerouting does not cancel it.
    Participant departure and worker shutdown cancel and drain owned work.
    Observations remain private to the job and never enter conversation context.
    """

    def __init__(
        self,
        frames: CurrentFrameTool,
        images: ImageRegistry,
        decisions: DecisionService,
        *,
        vlm: VLMService,
        llm: LLMService,
        context: ContextManager,
        runtime: AgentRuntime,
        output_topic: Topic[ResponseOutput],
        check_interval_secs: float = 1.0,
        confirm_interval_secs: float = 3.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create the agent.

        :param llm: Model that compiles an alert request into a watch.
        :param context: Participant context read, never written, when compiling.
        :param check_interval_secs: Minimum time between screening checks of consecutive frames.
        :param confirm_interval_secs: Minimum time between checks of a frame the screen passed,
            which run Cosmos, so repeated near-misses are paced more slowly.
        :param sleep: Pacing delay; tests replace it.
        """

        if check_interval_secs < 0 or confirm_interval_secs < check_interval_secs:
            raise ValueError("alert intervals must satisfy 0 <= check_interval_secs <= confirm_interval_secs")
        self._frames = frames
        self._images = images
        self._decisions = decisions
        self._vlm = vlm
        self._llm = llm
        self._context = context
        self._checks = AlertChecks(decisions, vlm)
        self._outputs = ResponseOutputs(runtime, output_topic, source="alerts", closed_errors=(RuntimeClosedError,))
        self._check_interval_secs = check_interval_secs
        self._confirm_interval_secs = confirm_interval_secs
        self._sleep = sleep
        self._jobs: dict[str, _Job] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, tuple[str, str]] = {}
        self._closing = False

    # RoutableAgent contract.

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid("alerts", _ROUTING, eligibility="active" if participant_id in self._jobs else "available")

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        del focused
        self._pending.pop(turn.participant_id, None)
        if self._closing:
            return Admission("decline")
        action = await self._classify(turn)
        if self._closing or action == "decline":
            return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, action)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        action = self._take_pending(turn)
        if action is None:
            return _NO_ANSWER
        participant_id = turn.participant_id
        async with self._lock(participant_id):
            if self._closing:
                return _NO_ANSWER
            if action == "cancel":
                await self._retire(participant_id)
                return await self._reply(turn, started, _CANCELLED)
            # Compile under the lock so a departure or shutdown waits for it and then cancels the new job.
            alert = await self._compile(turn)
            if self._closing:
                return _NO_ANSWER
            if alert is None:
                # An unresolvable request leaves any running alert alone.
                return await self._reply(turn, started, _UNCLEAR)
            await self._retire(participant_id)
            result = await self._reply(turn, started, f"I'll alert you once when {alert.condition}.")
            if not self._closing:
                self._start_job(participant_id, alert)
            return result

    async def release(self, participant_id: str) -> None:
        """Keep an independently requested alert running when another route answers."""

    async def participant_left(self, participant_id: str) -> None:
        async with self._lock(participant_id):
            self._pending.pop(participant_id, None)
            await self._cancel(participant_id)

    async def stop(self) -> None:
        self._closing = True
        for participant_id in tuple(self._jobs):
            await self.participant_left(participant_id)
        self._pending.clear()

    # The background watch.

    async def _watch(self, participant_id: str, job: _Job) -> None:
        """Wait for a frame that confirms the condition, announce it, then describe that frame."""

        try:
            trigger = await self._wait_for_trigger(participant_id, job)
            if await self._announce(participant_id, job, _ALERT):
                await self._describe(participant_id, job, trigger)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- isolate a failed background job and report its retirement.
            logger.exception("visual alert stopped participant={!r}", participant_id)
            await self._announce(participant_id, job, _STOPPED)
        finally:
            # Identity prevents retired work from removing a replacement job.
            if self._jobs.get(participant_id) is job:
                del self._jobs[participant_id]

    async def _wait_for_trigger(self, participant_id: str, job: _Job) -> Trigger:
        while True:
            started = time.monotonic()
            image = await self._current_image(participant_id)
            # Only a frame that may show the subject pays for Cosmos and the condition checks.
            screened = await self._checks.screen(participant_id, job.alert, image)
            if screened and (trigger := await self._checks.confirm(participant_id, job.alert, image)):
                return trigger
            await self._pace(started, screened=screened)

    async def _current_image(self, participant_id: str) -> ImageInput:
        frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
        return self._images.resolve(frame.image)

    async def _pace(self, started: float, *, screened: bool) -> None:
        """Wait out the rest of the interval; a screened frame ran Cosmos, so it waits longer."""

        interval = self._confirm_interval_secs if screened else self._check_interval_secs
        await self._sleep(max(0.0, interval - (time.monotonic() - started)))

    async def _announce(self, participant_id: str, job: _Job, text: str) -> bool:
        """Publish one urgent update while ``job`` is still current; never retry a failed delivery."""

        async with self._lock(participant_id):
            if self._jobs.get(participant_id) is not job:
                return False
            try:
                await self._outputs.update(participant_id, text, urgent=True, timestamp_us=time.time_ns() // 1000)
            except Exception:  # noqa: BLE001 -- log delivery failure without a duplicate publication.
                # Publication may have partially delivered; never retry or turn it into another message.
                logger.exception("failed to publish visual alert participant={!r}", participant_id)
                return False
            return True

    async def _describe(self, participant_id: str, job: _Job, trigger: Trigger) -> None:
        """Follow the announcement with one description of the triggering frame."""

        # Called directly, not through StreamingImageQueryTool, so cancelling the job stops the
        # model stream before departure or shutdown returns (NVIDIA/xr-ai#697).
        stream = self._vlm.stream_images([trigger.image], job.alert.condition, system_prompt=_DESCRIPTION)
        try:
            text = "".join([chunk async for chunk in stream if chunk])
            # An update is finite, so the description is spoken once it is complete.
            if self._jobs.get(participant_id) is job:
                await self._outputs.update(participant_id, text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the initial alert has already been delivered.
            logger.exception("visual alert description failed participant={!r}", participant_id)
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 -- cleanup must not repeat an already published alert.
                logger.exception("failed to close alert description participant={!r}", participant_id)

    # Request handling and job ownership.

    async def _classify(self, turn: AgentTurn) -> str:
        state = {"request": turn.query, "alert_active": turn.participant_id in self._jobs}
        answer = await decide(
            self._decisions, "alert-control", "alert_action", _CONTROL, state, participant_id=turn.participant_id
        )
        return answer.choice

    def _take_pending(self, turn: AgentTurn) -> str | None:
        """Return the action admitted for this turn, if admission was for this turn."""

        pending = self._pending.pop(turn.participant_id, None)
        return pending[1] if pending is not None and pending[0] == turn.turn_id else None

    async def _compile(self, turn: AgentTurn) -> CompiledAlert | None:
        context = "\n\n".join(block.text for block in self._context.blocks(turn.participant_id))
        with nemo_relay.scope.scope(
            "conversation-example.alert-compile",
            nemo_relay.ScopeType.Agent,
            input={"request": turn.query},
            metadata={"participant_id": turn.participant_id, "turn_id": turn.turn_id},
        ) as handle:
            alert = await compile_alert(self._llm, turn.query, history=turn.history, context=context)
            nemo_relay.scope.event(
                "conversation-example.alert-compile-result", handle=handle, data={"ready": alert is not None}
            )
        return alert

    async def _reply(self, turn: AgentTurn, started: Callable[[], Awaitable[None]], text: str) -> DestinationResult:
        async with self._outputs.reply(
            turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us, started=started
        ) as reply:
            await reply.send(text)
        return DestinationResult("answer", text)

    def _start_job(self, participant_id: str, alert: CompiledAlert) -> None:
        job = self._jobs[participant_id] = _Job(alert)
        job.task = asyncio.create_task(
            self._watch(participant_id, job),
            name=f"conversation-example-alert:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )

    async def _retire(self, participant_id: str) -> None:
        """Stop the current alert and take back any of its updates not yet spoken.

        A finished job has already left ``_jobs``, but its description can still be queued.
        """

        await self._cancel(participant_id)
        await self._outputs.withdraw(participant_id)

    async def _cancel(self, participant_id: str) -> None:
        job = self._jobs.pop(participant_id, None)
        if job is None or job.task is None:
            return
        job.task.cancel()
        await wait_despite_cancellation({job.task})

    def _lock(self, participant_id: str) -> asyncio.Lock:
        return self._locks.setdefault(participant_id, asyncio.Lock())
