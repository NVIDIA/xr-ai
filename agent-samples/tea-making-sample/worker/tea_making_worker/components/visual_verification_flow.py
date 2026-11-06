# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run independent visual verification until enough distinct images confirm a goal.

The caller supplies acquisition, typed services, policy and a state callback.
This coroutine owns its observation loop and evidence window. It knows nothing
about SOP sequencing, conversation, YAML or recording. Cancellation propagates
through its awaited operations; there are no detached observation tasks.
"""

from __future__ import annotations

import asyncio
import math
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from xr_ai_models import DecisionQuestion, DecisionService
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

_ObservationData = TypeVar("_ObservationData", bound=BaseModel)

Judgment = Literal["positive", "negative", "unknown"]


@dataclass(frozen=True, slots=True)
class VisualObservation:
    """One original image answer and its judgment, retained as bounded context."""

    timestamp_us: int
    frame_key: tuple[str, int, int]
    description: str
    judgment: Judgment
    truncated: bool = False
    parsed: BaseModel | None = None


@dataclass(frozen=True, slots=True)
class VisualVerificationState:
    """Complete current verification state; unknowns stay outside the evidence window."""

    status: Literal["running", "completed"]
    question: str
    completion_criteria: dict[str, str]
    started_at_us: int
    updated_at_us: int
    observations: tuple[VisualObservation, ...]
    confirmation_window: tuple[Judgment, ...]
    required_positives: int
    window_size: int
    observations_omitted: int
    positives_in_window: int
    valid_observations_in_window: int
    waiting_reason: str = ""


async def run_visual_verification_flow(
    *,
    frame_source: Callable[[], Awaitable[ImageFrame | None]],
    observer: ImageQueryTool,
    decisions: DecisionService,
    question: str,
    completion: DecisionQuestion,
    required_positives: int,
    window_size: int,
    poll_interval_seconds: float,
    history_limit: int,
    observation_text_limit: int,
    timeout_seconds: float,
    history_max_age_seconds: float | None = None,
    on_state: Callable[[VisualVerificationState], Awaitable[None]] | None = None,
    observation_model: type[_ObservationData] | None = None,
    can_assess: Callable[[_ObservationData], bool] | None = None,
) -> VisualVerificationState:
    """Observe fresh frames, judge evidence with Clef and return at the threshold.

    Args:
        frame_source: Caller-owned acquisition returning a frame, or None while unavailable.
        observer: Typed image tool using the caller's Omni/system instructions.
        decisions: Choice service; completion must define positive/negative/unknown.
        question: Focused image question built by the caller from any earlier results.
        completion: Instructions and three criteria defining completion evidence.
        required_positives: Positive observations needed within the evidence window.
        window_size: Maximum number of decidable observations in the sliding window.
        poll_interval_seconds: Delay between acquisition attempts.
        history_limit: Maximum recent observations retained in each state snapshot.
        observation_text_limit: Maximum characters retained per observation; truncation is labeled.
        timeout_seconds: Timeout for each acquisition/inference operation.
        history_max_age_seconds: Optional age limit for contextual observation history.
        observation_model: Optional native schema for the visual data instance. Invalid output remains unknown.
        can_assess: Optional deterministic readiness predicate over parsed data; false preserves
            the partial observation as unknown without calling the completion service. Requires a schema.
        on_state: Async callback awaited in this coroutine after each distinct observation.
            Exceptions propagate to the owning runner; cancellation ends further callbacks.

    Returns:
        The completed state, including its evidence window and bounded original context.

    Raises:
        ModelServiceError: An inference service fails. The runner owns failure policy.

    Unknown evidence is retained for context but consumes no confirmation slot.
    Re-reading an image cannot confirm a condition twice. Acquisition/model errors
    propagate rather than being silently turned into successful or negative evidence.
    """
    if observation_model is not None and (not isinstance(observation_model, type) or not issubclass(observation_model, BaseModel)):
        raise TypeError("observation_model must be a Pydantic model type")
    if can_assess is not None and (observation_model is None or not callable(can_assess)):
        raise ValueError("can_assess requires observation_model and a callable predicate")
    observation_question = question
    if observation_model is not None:
        observation_question += "\nReturn only a JSON data instance matching this schema, not the schema itself:\n" + json.dumps(observation_model.model_json_schema())
    if not question.strip() or set(completion.criteria) != {"positive", "negative", "unknown"}:
        raise ValueError("visual verification requires a question and positive/negative/unknown criteria")
    if type(required_positives) is not int or type(window_size) is not int:
        raise ValueError("confirmation bounds must be integers")
    if type(history_limit) is not int or type(observation_text_limit) is not int:
        raise ValueError("history bounds must be integers")
    if not 1 <= required_positives <= window_size:
        raise ValueError("required positives must fit in the confirmation window")
    if (
        not math.isfinite(poll_interval_seconds)
        or not math.isfinite(timeout_seconds)
        or min(poll_interval_seconds, history_limit, observation_text_limit, timeout_seconds) <= 0
    ):
        raise ValueError("visual loop intervals and context bounds must be positive")
    if history_max_age_seconds is not None and (
        not math.isfinite(history_max_age_seconds) or history_max_age_seconds <= 0
    ):
        raise ValueError("history age must be positive when configured")

    started_at_us = time.time_ns() // 1000
    observations: deque[VisualObservation] = deque()
    evidence_window: deque[Judgment] = deque(maxlen=window_size)
    # Per-track high-water marks reject old/duplicate images without retaining an
    # unbounded set of every image seen during a long pause.
    latest_frame_keys: dict[str, tuple[int, int]] = {}
    omitted_count = 0

    while True:
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        async with asyncio.timeout(timeout_seconds):
            frame = await frame_source()
        if frame is None:
            while (
                history_max_age_seconds is not None
                and observations
                and time.time_ns() // 1000 - observations[0].timestamp_us > history_max_age_seconds * 1_000_000
            ):
                observations.popleft()
                omitted_count += 1
            if on_state is not None:
                await on_state(
                    VisualVerificationState(
                        "running",
                        question,
                        dict(completion.criteria),
                        started_at_us,
                        time.time_ns() // 1000,
                        tuple(observations),
                        tuple(evidence_window),
                        required_positives,
                        window_size,
                        omitted_count,
                        evidence_window.count("positive"),
                        len(evidence_window),
                        "A current camera frame is unavailable.",
                    )
                )
            await asyncio.sleep(poll_interval_seconds)
            continue
        frame_position = (frame.timestamp_us, frame.sequence)
        if frame_position <= latest_frame_keys.get(frame.track_id, (-1, -1)):
            await asyncio.sleep(poll_interval_seconds)
            continue
        latest_frame_keys[frame.track_id] = frame_position

        async with asyncio.timeout(timeout_seconds):
            answer = await observer.execute(ImageQueryRequest(image=frame.image, query=observation_question))
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        description = answer.text
        judgment: Judgment = "unknown"
        parsed = None
        waiting_reason = ""
        ready_to_assess = answer.available and bool(description.strip())
        if ready_to_assess and observation_model is not None:
            try:
                parsed = observation_model.model_validate_json(description)
            except ValidationError:
                ready_to_assess = False
                waiting_reason = "Visual output does not match the observation schema."
            else:
                if can_assess is not None and not can_assess(parsed):
                    ready_to_assess = False
                    waiting_reason = "The observation is partial; completion cannot yet be assessed."
        if ready_to_assess:
            verdict = await decisions.decide(
                {"observation": parsed.model_dump(mode="json") if parsed is not None else description},
                {"completion": completion},
                timeout=timeout_seconds,
            )
            judgment = verdict.answers["completion"].choice
            if judgment not in completion.criteria:
                raise ValueError("visual judgment is outside the supplied criteria")

        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        sampled_at_us = time.time_ns() // 1000
        truncated = len(description) > observation_text_limit
        observations.append(
            VisualObservation(
                sampled_at_us,
                (frame.track_id, frame.sequence, frame.timestamp_us),
                description[:observation_text_limit],
                judgment,
                truncated,
                parsed.model_copy(deep=True) if parsed is not None else None,
            )
        )
        while len(observations) > history_limit or (
            history_max_age_seconds is not None
            and observations
            and sampled_at_us - observations[0].timestamp_us > history_max_age_seconds * 1_000_000
        ):
            observations.popleft()
            omitted_count += 1
        if judgment != "unknown":
            evidence_window.append(judgment)
        complete = evidence_window.count("positive") >= required_positives
        state = VisualVerificationState(
            "completed" if complete else "running",
            question,
            dict(completion.criteria),
            started_at_us,
            sampled_at_us,
            tuple(observations),
            tuple(evidence_window),
            required_positives,
            window_size,
            omitted_count,
            evidence_window.count("positive"),
            len(evidence_window),
            waiting_reason,
        )
        if on_state is not None:
            await on_state(state)
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if complete:
            return state
        await asyncio.sleep(poll_interval_seconds)
