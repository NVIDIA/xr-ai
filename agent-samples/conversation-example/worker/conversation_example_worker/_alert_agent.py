# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own participant-scoped, one-shot background visual alerts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_conversation import ContextManager, ConversationContextUpdate
from xr_ai_interaction import ResponseOutput, ResponseOutputs
from xr_ai_models import DecisionQuestion, DecisionService, ImageInput, LLMService, VLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutableAgent, RoutingAid
from xr_ai_runtime import AgentRuntime, RuntimeClosedError, Topic
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool, StreamingImageQueryTool

from ._alert_compile import CompiledAlert, compile_alert

ALERT_CAPABILITY_TOPIC = Topic("conversation-example.alert-capability", ConversationContextUpdate)
_CAPABILITY = "Capability: alert you once when a requested visual condition is seen, or cancel a pending visual alert."
_PROMPTS = Path(__file__).parent / "prompts"
_CONTROL = (_PROMPTS / "alert_control.txt").read_text(encoding="utf-8").strip()
_RELEVANCE = (_PROMPTS / "alert_relevance.txt").read_text(encoding="utf-8").strip()
_CONDITION = (_PROMPTS / "alert_condition.txt").read_text(encoding="utf-8").strip()
_DESCRIPTION = (_PROMPTS / "alert_description.txt").read_text(encoding="utf-8").strip()
_BACKUP_PROMPT = (
    "Describe only visible evidence relevant to the requested alert condition. Say when evidence is absent, "
    "unclear, or obscured. Use this single current still image only. Do not infer motion, an event, a change "
    "over time, a previous state, or anything outside this image. Be concise and factual."
)


@dataclass(frozen=True)
class _Trigger:
    image: ImageInput
    evidence: str
    path: str
    confidence: float


@dataclass
class _Job:
    condition: str
    relevance: str
    response_id: str
    task: asyncio.Task[None] | None = None
    checks: set[asyncio.Task[_Trigger | None]] = field(default_factory=set)


class VisualAlertAgent(RoutableAgent):
    """Watch one condition per participant; announce and describe its triggering frame.

    A request is first compiled with the participant's recent exchanges and saved context into a
    standalone condition, so an alert can refer to something said or noted earlier. Each watched
    frame passes a cheap Clef relevance screen before the costly check: only a relevant frame is
    judged against the condition, by a direct Clef image decision and by a Cosmos observation
    judged by Clef, whichever confirms first. Checks are paced by interval.

    Only an explicit new alert replaces a job. Rerouting does not cancel it.
    Participant departure and worker shutdown cancel and drain owned work.
    Observations remain private to the job and never enter conversation context.
    """

    def __init__(
        self,
        frames: CurrentFrameTool,
        images: ImageRegistry,
        decisions: DecisionService,
        publish_capability: Callable[[str, ConversationContextUpdate], Awaitable[None]],
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
        self._vlm = vlm
        self._llm = llm
        self._context = context
        self._check_interval_secs = check_interval_secs
        self._confirm_interval_secs = confirm_interval_secs
        self._sleep = sleep
        self._frames = frames
        self._images = images
        self._decisions = decisions
        self._publish_capability = publish_capability
        self._outputs = ResponseOutputs(runtime, output_topic, source="alerts", closed_errors=(RuntimeClosedError,))
        self._jobs: dict[str, _Job] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, tuple[str, str]] = {}
        self._closing = False

    def routing_aid(self, participant_id: str) -> RoutingAid:
        active = participant_id in self._jobs
        return RoutingAid(
            "alerts",
            (
                "Set a one-shot background visual alert ONLY when explicitly asked to notify the user "
                "when a specific visible condition occurs, including a condition they gave earlier or "
                "noted; cancel an existing alert on an explicit "
                "request. Visual questions, observations, recall, and fragments are not alert "
                "controls."
            ),
            eligibility="active" if active else "available",
        )

    async def participant_joined(self, participant_id: str) -> None:
        """Publish static capability only, without job state or observations."""
        await self._publish_capability(participant_id, ConversationContextUpdate(text=_CAPABILITY))

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        del focused
        self._pending.pop(turn.participant_id, None)
        if self._closing:
            return Admission("decline")
        question = DecisionQuestion(
            instructions=_CONTROL,
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
        with nemo_relay.scope.scope(
            "conversation-example.alert-control",
            nemo_relay.ScopeType.Agent,
            input={"request": turn.query, "alert_active": turn.participant_id in self._jobs},
            metadata={"participant_id": turn.participant_id, "turn_id": turn.turn_id},
        ) as scope:
            response = await self._decisions.decide(
                {"request": turn.query, "alert_active": turn.participant_id in self._jobs}, {"alert_action": question}
            )
            action = response.answers["alert_action"].choice
            if action not in question.criteria:
                raise ValueError(f"Clef returned invalid alert action {action!r}")
            nemo_relay.scope.event(
                "conversation-example.alert-control-result",
                handle=scope,
                data={"choice": action, "confidence": response.answers["alert_action"].confidence},
            )
        if self._closing or action == "decline":
            return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, action)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)
        if pending is None or pending[0] != turn.turn_id:
            return DestinationResult("answer", "")
        action = pending[1]
        async with self._locks.setdefault(turn.participant_id, asyncio.Lock()):
            if self._closing:
                return DestinationResult("answer", "")
            # Compile under the lock so a departure or shutdown waits for it and then cancels the new job.
            compiled = await self._compile(turn) if action == "start" else None
            if self._closing:
                return DestinationResult("answer", "")
            if action == "start" and compiled is None:
                # An unresolvable request leaves any running alert alone.
                text = "I couldn't tell what to watch for. Please say what should trigger the alert."
                async with self._outputs.reply(
                    turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us, started=started
                ) as reply:
                    await reply.send(text)
                return DestinationResult("answer", text)
            await self._cancel(turn.participant_id)
            text = (
                "The visual alert is cancelled."
                if compiled is None
                else f"I'll alert you once when {compiled.condition}."
            )
            async with self._outputs.reply(
                turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us, started=started
            ) as reply:
                await reply.send(text)
            if compiled is not None and not self._closing:
                job = _Job(compiled.condition, compiled.relevance, f"alert:{turn.turn_id}")
                self._jobs[turn.participant_id] = job
                job.task = asyncio.create_task(
                    self._watch(turn.participant_id, job),
                    name=f"conversation-example-alert:{turn.participant_id}",
                    context=nemo_relay.fork_asyncio_context(),
                )
            return DestinationResult("answer", text)

    async def release(self, participant_id: str) -> None:
        """Keep an independently requested alert running when another route answers."""

    async def participant_left(self, participant_id: str) -> None:
        async with self._locks.setdefault(participant_id, asyncio.Lock()):
            self._pending.pop(participant_id, None)
            await self._cancel(participant_id)

    async def stop(self) -> None:
        self._closing = True
        for participant_id in tuple(self._jobs):
            await self.participant_left(participant_id)
        self._pending.clear()

    async def _cancel(self, participant_id: str) -> None:
        job = self._jobs.pop(participant_id, None)
        if job is not None and job.task is not None:
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)

    async def _compile(self, turn: AgentTurn) -> CompiledAlert | None:
        context = "\n\n".join(block.text for block in self._context.blocks(turn.participant_id))
        with nemo_relay.scope.scope(
            "conversation-example.alert-compile",
            nemo_relay.ScopeType.Agent,
            input={"request": turn.query},
            metadata={"participant_id": turn.participant_id, "turn_id": turn.turn_id},
        ) as scope:
            compiled = await compile_alert(self._llm, turn.query, history=turn.history, context=context)
            nemo_relay.scope.event(
                "conversation-example.alert-compile-result",
                handle=scope,
                data={"ready": compiled is not None},
            )
        return compiled

    async def _watch(self, participant_id: str, job: _Job) -> None:
        try:
            while self._jobs.get(participant_id) is job:
                started = time.monotonic()
                frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
                if self._jobs.get(participant_id) is not job:
                    return
                image = self._images.resolve(frame.image)
                # Only a frame that may show the subject pays for the Cosmos and condition checks.
                screened = await self._screen(participant_id, job, image)
                trigger = await self._confirm(participant_id, job, image) if screened else None
                if self._jobs.get(participant_id) is not job:
                    return
                if trigger is not None:
                    with nemo_relay.scope.scope(
                        "conversation-example.alert-triggered",
                        nemo_relay.ScopeType.Agent,
                        input={
                            "path": trigger.path,
                            "confidence": trigger.confidence,
                            "evidence": trigger.evidence,
                        },
                        metadata={"participant_id": participant_id},
                    ) as scope:
                        nemo_relay.scope.event(
                            "conversation-example.alert-triggered-result",
                            handle=scope,
                            data={"path": trigger.path, "confidence": trigger.confidence},
                        )
                    if await self._publish_notice(participant_id, job, "Alert"):
                        await self._describe(participant_id, job, trigger.image)
                    return
                interval = self._confirm_interval_secs if screened else self._check_interval_secs
                await self._sleep(max(0.0, interval - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- isolate a failed background job and report its retirement.
            logger.exception("visual alert stopped participant={!r}", participant_id)
            await self._publish_notice(participant_id, job, "I couldn't check the visual alert. It has stopped.")
        finally:
            await asyncio.gather(*job.checks, return_exceptions=True)
            # Identity prevents retired work from removing a replacement job; no await during cleanup.
            if self._jobs.get(participant_id) is job:
                self._jobs.pop(participant_id, None)

    async def _screen(self, participant_id: str, job: _Job, image: ImageInput) -> bool:
        question = DecisionQuestion(
            instructions=_RELEVANCE,
            criteria={
                "yes": "The subject of the alert may be visible in the image, even partly or uncertainly.",
                "no": "The subject of the alert is clearly absent from the image.",
            },
        )
        state = {"condition": job.condition, "relevance": job.relevance}
        with nemo_relay.scope.scope(
            "conversation-example.alert-relevance",
            nemo_relay.ScopeType.Agent,
            input=state,
            metadata={"participant_id": participant_id},
        ) as scope:
            response = await self._decisions.decide(
                {"decision_path": "relevance", **state}, {"relevant": question}, images=[image]
            )
            answer = response.answers["relevant"]
            if answer.choice not in question.criteria:
                raise ValueError(f"Clef returned invalid alert relevance result {answer.choice!r}")
            nemo_relay.scope.event(
                "conversation-example.alert-relevance-result",
                handle=scope,
                data={"choice": answer.choice, "confidence": answer.confidence},
            )
        return answer.choice == "yes"

    async def _confirm(self, participant_id: str, job: _Job, image: ImageInput) -> _Trigger | None:
        """Judge one screened frame; the first path to confirm the condition wins."""

        # The job owns the checks so a confirmed alert is announced without waiting for the other
        # check to finish cancelling; the watch drains them before it retires.
        checks = job.checks = {
            asyncio.create_task(
                check(participant_id, job, image),
                name=name,
                context=nemo_relay.fork_asyncio_context(),
            )
            for name, check in (
                ("alert-fast-clef", self._confirm_direct),
                ("alert-cosmos-clef", self._confirm_observed),
            )
        }
        failures = 0
        try:
            while checks:
                done, _ = await asyncio.wait(checks, return_when=asyncio.FIRST_COMPLETED)
                for check in done:
                    checks.discard(check)
                    error = asyncio.CancelledError() if check.cancelled() else check.exception()
                    if error is not None:
                        failures += 1
                        logger.warning(
                            "visual alert check failed participant={!r} path={!r}: {}",
                            participant_id,
                            check.get_name(),
                            error,
                        )
                    elif (trigger := check.result()) is not None:
                        return trigger
            # One failed path leaves the other usable for later frames; both failing retires the job.
            if failures == 2:
                raise RuntimeError("all visual alert checks failed")
            return None
        finally:
            for check in checks:
                check.cancel()

    async def _confirm_direct(self, participant_id: str, job: _Job, image: ImageInput) -> _Trigger | None:
        question = DecisionQuestion(
            instructions=_CONDITION,
            criteria={
                "yes": "The image clearly and directly establishes the requested visible condition.",
                "no": "The condition is absent, uncertain, obscured, or cannot be established from this single image.",
            },
        )
        return await self._judge(
            participant_id,
            path="fast-clef",
            state={"condition": job.condition},
            question=question,
            images=[image],
            trigger=_Trigger(image, "direct image decision", "fast-clef", 0.0),
        )

    async def _confirm_observed(self, participant_id: str, job: _Job, image: ImageInput) -> _Trigger | None:
        question = DecisionQuestion(
            instructions=(
                "Judge only whether the written observation establishes the requested visible condition. "
                "Use yes only when the observation directly supports it; otherwise use no. A single observation "
                "from one still image cannot establish motion, an event, a change over time, or a previous state; "
                "such conditions mean no."
            ),
            criteria={
                "yes": "The observation clearly establishes the requested visible condition.",
                "no": "The observation is absent, uncertain, obscured, or does not establish the condition.",
            },
        )
        held_images = ImageRegistry(capacity=1)
        reference = held_images.put(image)
        with nemo_relay.scope.scope(
            "conversation-example.alert-observation",
            nemo_relay.ScopeType.Agent,
            input={"condition": job.condition, "image_role": "current still frame"},
            metadata={"participant_id": participant_id},
        ) as scope:
            observation = await ImageQueryTool(
                images=held_images,
                vlm=self._vlm,
                system_prompt=_BACKUP_PROMPT,
            ).execute(ImageQueryRequest(image=reference, query=job.condition))
            nemo_relay.scope.event(
                "conversation-example.alert-observation-result",
                handle=scope,
                data={"available": observation.available},
            )
        if not observation.available:
            raise RuntimeError("Cosmos could not inspect the alert frame")
        return await self._judge(
            participant_id,
            path="cosmos-clef",
            state={"condition": job.condition, "current_observation": observation.text},
            question=question,
            images=(),
            trigger=_Trigger(image, observation.text, "cosmos-clef", 0.0),
        )

    async def _judge(
        self,
        participant_id: str,
        *,
        path: str,
        state: dict[str, str],
        question: DecisionQuestion,
        images: Sequence[ImageInput],
        trigger: _Trigger,
    ) -> _Trigger | None:
        with nemo_relay.scope.scope(
            "conversation-example.alert-condition",
            nemo_relay.ScopeType.Agent,
            input={"path": path, **state},
            metadata={"participant_id": participant_id},
        ) as scope:
            response = await self._decisions.decide(
                {"decision_path": path, **state}, {"condition_met": question}, images=images
            )
            answer = response.answers["condition_met"]
            if answer.choice not in question.criteria:
                raise ValueError(f"Clef returned invalid alert condition result {answer.choice!r}")
            nemo_relay.scope.event(
                "conversation-example.alert-condition-result",
                handle=scope,
                data={"path": path, "choice": answer.choice, "confidence": answer.confidence},
            )
        if answer.choice != "yes":
            return None
        return _Trigger(trigger.image, trigger.evidence, path, answer.confidence)

    async def _publish_notice(self, participant_id: str, job: _Job, text: str) -> bool:
        if self._jobs.get(participant_id) is not job:
            return False
        async with self._locks.setdefault(participant_id, asyncio.Lock()):
            if self._jobs.get(participant_id) is not job:
                return False
            try:
                await self._outputs.update(participant_id, text, urgent=True, timestamp_us=time.time_ns() // 1000)
            except Exception:  # noqa: BLE001 -- log delivery failure without a duplicate publication.
                # Publication may have partially delivered; never retry or turn it into another message.
                logger.exception("failed to publish visual alert participant={!r}", participant_id)
                return False
            return True

    async def _describe(self, participant_id: str, job: _Job, image: ImageInput) -> None:
        # Hold the selected pixels independently of the shared registry's bounded LRU.
        images = ImageRegistry(capacity=1)
        reference = images.put(image)
        tool = StreamingImageQueryTool(images=images, vlm=self._vlm, system_prompt=_DESCRIPTION)
        stream = tool.stream(ImageQueryRequest(image=reference, query=job.condition))
        fragments: list[str] = []
        try:
            async for chunk in stream:
                if self._jobs.get(participant_id) is not job:
                    return
                if chunk.text:
                    fragments.append(chunk.text)
            # An update is finite, so the description is spoken once it is complete.
            await self._outputs.update(participant_id, "".join(fragments))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the initial alert has already been delivered.
            logger.exception("visual alert description failed participant={!r}", participant_id)
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 -- cleanup must not repeat an already published alert.
                logger.exception("failed to close alert description participant={!r}", participant_id)
