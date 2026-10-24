# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge one watched frame against a compiled alert.

A frame first passes a cheap Clef relevance screen. Only a relevant frame is
confirmed, by two checks raced on that frame: a direct Clef image decision, and
a Cosmos observation judged by Clef. The first check to confirm wins.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nemo_relay
from loguru import logger
from xr_ai_models import DecisionQuestion, DecisionService, ImageInput, VLMService
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from ._alert_compile import CompiledAlert
from ._decisions import decide
from ._tasks import wait_despite_cancellation

_PROMPTS = Path(__file__).parent / "prompts"
_RELEVANCE = DecisionQuestion(
    instructions=(_PROMPTS / "alert_relevance.txt").read_text(encoding="utf-8").strip(),
    criteria={
        "yes": "The subject of the alert may be visible in the image, even partly or uncertainly.",
        "no": "The subject of the alert is clearly absent from the image.",
    },
)
_DIRECT = DecisionQuestion(
    instructions=(_PROMPTS / "alert_condition.txt").read_text(encoding="utf-8").strip(),
    criteria={
        "yes": "The image clearly and directly establishes the requested visible condition.",
        "no": "The condition is absent, uncertain, obscured, or cannot be established from this single image.",
    },
)
_OBSERVED = DecisionQuestion(
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
_OBSERVATION_PROMPT = (
    "Describe only visible evidence relevant to the requested alert condition. Say when evidence is absent, "
    "unclear, or obscured. Use this single current still image only. Do not infer motion, an event, a change "
    "over time, a previous state, or anything outside this image. Be concise and factual."
)


@dataclass(frozen=True, slots=True)
class Trigger:
    """A frame that confirmed the alert condition."""

    image: ImageInput
    evidence: str
    path: str
    confidence: float


class AlertChecks:
    """Screen and confirm frames for compiled alerts."""

    def __init__(self, decisions: DecisionService, vlm: VLMService) -> None:
        self._decisions = decisions
        self._vlm = vlm

    async def screen(self, participant_id: str, alert: CompiledAlert, image: ImageInput) -> bool:
        """Return whether the alert's subject may be in ``image``."""

        state = {"decision_path": "relevance", "condition": alert.condition, "relevance": alert.relevance}
        answer = await decide(
            self._decisions,
            "alert-relevance",
            "relevant",
            _RELEVANCE,
            state,
            participant_id=participant_id,
            images=[image],
        )
        return answer.choice == "yes"

    async def confirm(self, participant_id: str, alert: CompiledAlert, image: ImageInput) -> Trigger | None:
        """Race both confirmations on ``image`` and return the first trigger.

        Both checks finish, including cancellation cleanup, before this returns or
        propagates cancellation. One failed check leaves the other's verdict.

        :raises RuntimeError: Both checks failed.
        """

        trigger = await _first_trigger(
            participant_id,
            {
                "alert-fast-clef": self._confirm_directly(participant_id, alert, image),
                "alert-cosmos-clef": self._confirm_by_observation(participant_id, alert, image),
            },
        )
        if trigger is not None:
            _trace_trigger(participant_id, trigger)
        return trigger

    async def _confirm_directly(self, participant_id: str, alert: CompiledAlert, image: ImageInput) -> Trigger | None:
        state = {"decision_path": "fast-clef", "condition": alert.condition}
        answer = await decide(
            self._decisions,
            "alert-condition",
            "condition_met",
            _DIRECT,
            state,
            participant_id=participant_id,
            images=[image],
        )
        return (
            Trigger(image, "direct image decision", "fast-clef", answer.confidence) if answer.choice == "yes" else None
        )

    async def _confirm_by_observation(
        self, participant_id: str, alert: CompiledAlert, image: ImageInput
    ) -> Trigger | None:
        observation = await self._observe(participant_id, alert, image)
        state = {"decision_path": "cosmos-clef", "condition": alert.condition, "current_observation": observation}
        answer = await decide(
            self._decisions, "alert-condition", "condition_met", _OBSERVED, state, participant_id=participant_id
        )
        return Trigger(image, observation, "cosmos-clef", answer.confidence) if answer.choice == "yes" else None

    async def _observe(self, participant_id: str, alert: CompiledAlert, image: ImageInput) -> str:
        """Ask Cosmos what ``image`` shows about the condition."""

        # Hold the frame independently of the shared registry's bounded LRU.
        held = ImageRegistry(capacity=1)
        request = ImageQueryRequest(image=held.put(image), query=alert.condition)
        with nemo_relay.scope.scope(
            "conversation-example.alert-observation",
            nemo_relay.ScopeType.Agent,
            input={"condition": alert.condition, "image_role": "current still frame"},
            metadata={"participant_id": participant_id},
        ) as handle:
            observation = await ImageQueryTool(images=held, vlm=self._vlm, system_prompt=_OBSERVATION_PROMPT).execute(
                request
            )
            nemo_relay.scope.event(
                "conversation-example.alert-observation-result",
                handle=handle,
                data={"available": observation.available},
            )
        if not observation.available:
            raise RuntimeError("Cosmos could not inspect the alert frame")
        return observation.text


async def _first_trigger(
    participant_id: str, checks: Mapping[str, Coroutine[Any, Any, Trigger | None]]
) -> Trigger | None:
    tasks = {
        asyncio.create_task(check, name=name, context=nemo_relay.fork_asyncio_context())
        for name, check in checks.items()
    }
    failures = 0
    try:
        pending = tasks
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = asyncio.CancelledError() if task.cancelled() else task.exception()
                if error is not None:
                    failures += 1
                    logger.warning(
                        "visual alert check failed participant={!r} path={!r}: {}",
                        participant_id,
                        task.get_name(),
                        error,
                    )
                elif (trigger := task.result()) is not None:
                    return trigger
    finally:
        # No check outlives this call: losers are cancelled and fully drained, even when the
        # watch itself is cancelled again during that cleanup.
        for task in tasks:
            task.cancel()
        await wait_despite_cancellation(tasks)
    if failures == len(checks):
        raise RuntimeError("all visual alert checks failed")
    return None


def _trace_trigger(participant_id: str, trigger: Trigger) -> None:
    with nemo_relay.scope.scope(
        "conversation-example.alert-triggered",
        nemo_relay.ScopeType.Agent,
        input={"path": trigger.path, "confidence": trigger.confidence, "evidence": trigger.evidence},
        metadata={"participant_id": participant_id},
    ) as handle:
        nemo_relay.scope.event(
            "conversation-example.alert-triggered-result",
            handle=handle,
            data={"path": trigger.path, "confidence": trigger.confidence},
        )
