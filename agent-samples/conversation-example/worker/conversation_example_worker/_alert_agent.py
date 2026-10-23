# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own participant-scoped, one-shot background visual alerts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_conversation import ConversationContextUpdate
from xr_ai_interaction import ResponseOutput
from xr_ai_models import DecisionQuestion, DecisionService, ImageInput, VLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutableAgent, RoutingAid
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool, StreamingImageQueryTool

ALERT_CAPABILITY_TOPIC = Topic("conversation-example.alert-capability", ConversationContextUpdate)
_CAPABILITY = "Capability: alert you once when a requested visual condition is seen, or cancel a pending visual alert."
_PROMPTS = Path(__file__).parent / "prompts"
_CONTROL = (_PROMPTS / "alert_control.txt").read_text(encoding="utf-8").strip()
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
    response_id: str
    task: asyncio.Task[None] | None = None
    winner: asyncio.Future[_Trigger] | None = None


class VisualAlertAgent(RoutableAgent):
    """Watch one condition per participant; announce and describe its triggering frame.

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
        runtime: AgentRuntime,
        output_topic: Topic[ResponseOutput],
    ) -> None:
        self._vlm = vlm
        self._frames = frames
        self._images = images
        self._decisions = decisions
        self._publish_capability = publish_capability
        self._runtime = runtime
        self._output_topic = output_topic
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
                "when a specific visible condition occurs; cancel an existing alert on an explicit "
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
                    "An explicit request to be alerted later when a specific visually observable condition is seen."
                ),
                "cancel": "An explicit request to cancel or stop a visual alert.",
                "decline": (
                    "Anything else, including current visual questions, observations, ambiguous fragments, "
                    "capability questions, or an unspecified alert condition."
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

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[bool]]) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)
        if pending is None or pending[0] != turn.turn_id:
            return DestinationResult("answer", "")
        action = pending[1]
        async with self._locks.setdefault(turn.participant_id, asyncio.Lock()):
            if self._closing:
                return DestinationResult("answer", "")
            await self._cancel(turn.participant_id)
            text = (
                "The visual alert is cancelled."
                if action == "cancel"
                else "I'll alert you once when that condition is visible."
            )
            interrupt = await started()
            await self._runtime.publish(
                self._output_topic,
                ResponseOutput(
                    text=text, response_id=turn.turn_id, interrupt=interrupt, timestamp_us=turn.timestamp_us
                ),
                participant_id=turn.participant_id,
                source="alerts",
            )
            if action == "start" and not self._closing:
                job = _Job(turn.query, f"alert:{turn.turn_id}")
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

    async def _watch(self, participant_id: str, job: _Job) -> None:
        winner: asyncio.Future[_Trigger] = asyncio.get_running_loop().create_future()
        job.winner = winner
        lanes = {
            asyncio.create_task(
                self._watch_fast(participant_id, job, winner),
                name="alert-fast-clef",
                context=nemo_relay.fork_asyncio_context(),
            ),
            asyncio.create_task(
                self._watch_backup(participant_id, job, winner),
                name="alert-cosmos-clef",
                context=nemo_relay.fork_asyncio_context(),
            ),
        }
        winner_wait: asyncio.Future[_Trigger] | None = None
        try:
            pending = set(lanes)
            winner_wait = asyncio.ensure_future(asyncio.shield(winner))
            while pending and not winner.done():
                done, _ = await asyncio.wait((*pending, winner_wait), return_when=asyncio.FIRST_COMPLETED)
                if winner.done():
                    break
                for lane in done:
                    if lane is winner_wait:
                        continue
                    pending.remove(lane)
                    error = asyncio.CancelledError() if lane.cancelled() else lane.exception()
                    if error is not None:
                        logger.warning(
                            "visual alert detector failed participant={!r} path={!r}: {}",
                            participant_id,
                            lane.get_name(),
                            error,
                        )
            if winner.done() and self._jobs.get(participant_id) is job:
                trigger = winner.result()
                for lane in lanes:
                    if not lane.done() and not lane.cancelling():
                        lane.cancel()
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
            elif self._jobs.get(participant_id) is job:
                logger.error("all visual alert detectors stopped participant={!r}", participant_id)
                await self._publish_notice(participant_id, job, "I couldn't check the visual alert. It has stopped.")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- isolate a failed background job and report its retirement.
            logger.exception("visual alert stopped participant={!r}", participant_id)
            await self._publish_notice(participant_id, job, "I couldn't check the visual alert. It has stopped.")
        finally:
            if winner_wait is not None and not winner_wait.done():
                winner_wait.cancel()
            for lane in lanes:
                if not lane.done() and not lane.cancelling():
                    lane.cancel()
            await asyncio.gather(*lanes, return_exceptions=True)
            # Identity prevents retired work from removing a replacement job; no await during cleanup.
            if self._jobs.get(participant_id) is job:
                self._jobs.pop(participant_id, None)

    async def _watch_fast(self, participant_id: str, job: _Job, winner: asyncio.Future[_Trigger]) -> None:
        question = DecisionQuestion(
            instructions=_CONDITION,
            criteria={
                "yes": "The image clearly and directly establishes the requested visible condition.",
                "no": "The condition is absent, uncertain, obscured, or cannot be established from this single image.",
            },
        )
        while self._jobs.get(participant_id) is job and not winner.done():
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
            if self._jobs.get(participant_id) is not job:
                return
            image = self._images.resolve(frame.image)
            await self._judge(
                participant_id,
                job,
                winner,
                path="fast-clef",
                state={"condition": job.condition},
                questions={"condition_met": question},
                images=[image],
                trigger=_Trigger(image, "direct image decision", "fast-clef", 0.0),
            )

    async def _watch_backup(self, participant_id: str, job: _Job, winner: asyncio.Future[_Trigger]) -> None:
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
        while self._jobs.get(participant_id) is job and not winner.done():
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
            if self._jobs.get(participant_id) is not job:
                return
            image = self._images.resolve(frame.image)
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
            if self._jobs.get(participant_id) is not job:
                return
            await self._judge(
                participant_id,
                job,
                winner,
                path="cosmos-clef",
                state={"condition": job.condition, "current_observation": observation.text},
                questions={"condition_met": question},
                images=(),
                trigger=_Trigger(image, observation.text, "cosmos-clef", 0.0),
            )

    async def _judge(
        self,
        participant_id: str,
        job: _Job,
        winner: asyncio.Future[_Trigger],
        *,
        path: str,
        state: dict[str, str],
        questions: dict[str, DecisionQuestion],
        images: Sequence[ImageInput],
        trigger: _Trigger,
    ) -> None:
        with nemo_relay.scope.scope(
            "conversation-example.alert-condition",
            nemo_relay.ScopeType.Agent,
            input={"path": path, **state},
            metadata={"participant_id": participant_id},
        ) as scope:
            response = await self._decisions.decide({"decision_path": path, **state}, questions, images=images)
            answer = response.answers["condition_met"]
            if answer.choice not in questions["condition_met"].criteria:
                raise ValueError(f"Clef returned invalid alert condition result {answer.choice!r}")
            confidence = answer.confidence
            nemo_relay.scope.event(
                "conversation-example.alert-condition-result",
                handle=scope,
                data={"path": path, "choice": answer.choice, "confidence": confidence},
            )
        if (
            answer.choice == "yes"
            and self._jobs.get(participant_id) is job
            and job.winner is winner
            and not winner.done()
        ):
            # Event-loop atomic claim: no await may separate the check from set_result.
            winner.set_result(_Trigger(trigger.image, trigger.evidence, path, confidence))

    async def _publish_notice(self, participant_id: str, job: _Job, text: str) -> bool:
        if self._jobs.get(participant_id) is not job:
            return False
        async with self._locks.setdefault(participant_id, asyncio.Lock()):
            if self._jobs.get(participant_id) is not job:
                return False
            try:
                await self._runtime.publish(
                    self._output_topic,
                    ResponseOutput(
                        text=text,
                        response_id=job.response_id,
                        final=True,
                        interrupt=True,
                        timestamp_us=time.time_ns() // 1000,
                    ),
                    participant_id=participant_id,
                    source="alerts",
                )
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
        response_id = f"{job.response_id}:description"
        opened = False
        delivery_failed = False
        try:
            async for chunk in stream:
                if self._jobs.get(participant_id) is not job:
                    return
                if not chunk.text:
                    continue
                try:
                    await self._runtime.publish(
                        self._output_topic,
                        ResponseOutput(text=chunk.text, response_id=response_id, final=False),
                        participant_id=participant_id,
                        source="alerts",
                    )
                except Exception:
                    delivery_failed = True
                    raise
                opened = True
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the initial alert has already been delivered.
            logger.exception("visual alert description failed participant={!r}", participant_id)
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 -- cleanup must not repeat an already published alert.
                logger.exception("failed to close alert description participant={!r}", participant_id)
            if opened and not delivery_failed:
                # Close a partial response on cancellation without adding more spoken text.
                try:
                    await self._runtime.publish(
                        self._output_topic,
                        ResponseOutput(response_id=response_id),
                        participant_id=participant_id,
                        source="alerts",
                    )
                except Exception:  # noqa: BLE001 -- do not retry partial output delivery.
                    logger.exception("failed to finalize alert description participant={!r}", participant_id)
