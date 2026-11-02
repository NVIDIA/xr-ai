# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract visual data and resolve requested fields using retrieval and generation.

Python callers provide visual/result models, prompts, retrieval tool and
query function. The flow observes until visual model validation succeeds, retrieves
source-supported fields, then generates only still-missing fields. It neither
reads application YAML nor writes SOP state. Partial validated results are emitted
before subsequent fallible calls so the caller can retain them when skipped.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.rag import RetrieveRequest, RetrieveResult
from xr_ai_tools.tools import Tool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

VisualData = TypeVar("VisualData", bound=BaseModel)
ResolvedData = TypeVar("ResolvedData", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class FieldSource:
    """Source of one validated field; generated values cannot claim a document quote."""

    kind: Literal["retrieved", "generated"]
    source: str = ""
    quote: str = ""


@dataclass(frozen=True, slots=True)
class VisionRAGRetrievalObservation:
    """A bounded original visual answer, including unusable evidence."""

    timestamp_us: int
    frame_key: tuple[str, int, int]
    description: str
    usable: bool
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class VisionRAGRetrievalState(Generic[VisualData, ResolvedData]):
    """Complete retrieval progress, validated data and bounded original evidence."""

    status: Literal["running", "completed"]
    phase: Literal["observing", "retrieving", "extracting", "generating", "complete"]
    goal: str
    started_at_us: int
    updated_at_us: int
    visual_data: VisualData | None
    resolved_data: ResolvedData
    sources: dict[str, FieldSource]
    observations: tuple[VisionRAGRetrievalObservation, ...]
    passages: tuple[dict[str, Any], ...]
    observations_omitted: int
    issue: str = ""


async def run_vision_rag_retrieval_flow(
    *,
    frame_source: Callable[[], Awaitable[ImageFrame | None]],
    observer: ImageQueryTool,
    llm: LLMService,
    retrieval: Tool[RetrieveRequest, RetrieveResult],
    visual_model: type[VisualData],
    result_model: type[ResolvedData],
    visual_question: str,
    extraction_instructions: str,
    fallback_instructions: str,
    retrieval_query: Callable[[VisualData], str],
    poll_interval_seconds: float,
    timeout_seconds: float,
    history_limit: int,
    observation_text_limit: int,
    max_retrieved_passages: int,
    retrieved_text_limit: int,
    history_max_age_seconds: float | None = None,
    on_state: Callable[[VisionRAGRetrievalState[VisualData, ResolvedData]], Awaitable[None]] | None = None,
) -> VisionRAGRetrievalState[VisualData, ResolvedData]:
    """Observe visual data, resolve requested fields and return the source-labeled result.

    Args:
        frame_source: Async acquisition; None means a view is not available yet.
        observer: Typed visual tool using caller-supplied image/system policy.
        llm: Language service for document extraction and missing-only generation.
        retrieval: Existing typed RAG tool; retrieval remains demonstrably separate.
        visual_model: Python model whose validation defines usable visual input.
        result_model: Python model with requested result fields defaulting to None.
        visual_question: Task wording; code appends the response schema.
        extraction_instructions: Source-extraction policy supplied by the application.
        fallback_instructions: Policy for generating recommendations for missing fields.
        retrieval_query: Python function connecting validated visual data to a retrieval query.
        poll_interval_seconds: Delay between observation/retry attempts.
        timeout_seconds: Timeout for acquisition/model/retrieval operations.
        history_limit: Maximum retained original visual observations.
        observation_text_limit: Maximum retained characters per observation.
        max_retrieved_passages: Retrieval and retained-passage limit.
        retrieved_text_limit: Maximum retained characters in each retrieved passage.
        history_max_age_seconds: Optional maximum age of contextual observations.
        on_state: Async full-state callback in this task; callback failures propagate.

    Returns:
        Complete visual/result state. Validated partial fields are reported
        in earlier snapshots even if the flow is later cancelled.

    All result fields must be optional so missing values are represented
    explicitly, not silently invented by model or code defaults.
    """
    if (
        min(
            poll_interval_seconds,
            timeout_seconds,
            history_limit,
            observation_text_limit,
            max_retrieved_passages,
            retrieved_text_limit,
        )
        <= 0
        or max_retrieved_passages > 20
    ):
        raise ValueError("visual extraction limits must be positive and retrieval count at most 20")
    if history_max_age_seconds is not None and history_max_age_seconds <= 0:
        raise ValueError("history age must be positive")
    empty_result = result_model.model_validate({})
    if any(value is not None for value in empty_result.model_dump().values()):
        raise ValueError("result fields must default to None; caller owns application defaults")
    started_at_us = time.time_ns() // 1000
    visual_data: VisualData | None = None
    resolved_data = empty_result
    sources: dict[str, FieldSource] = {}
    observations: deque[VisionRAGRetrievalObservation] = deque()
    passages: tuple[dict[str, Any], ...] = ()
    omitted_count = 0
    latest_frame_keys: dict[str, tuple[int, int]] = {}

    async def report(phase, *, complete=False, issue=""):
        """Publish fresh snapshots; never mutate an object already given to the runner."""
        nonlocal omitted_count
        while (
            history_max_age_seconds is not None
            and observations
            and time.time_ns() // 1000 - observations[0].timestamp_us > history_max_age_seconds * 1_000_000
        ):
            observations.popleft()
            omitted_count += 1
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        state = VisionRAGRetrievalState(
            "completed" if complete else "running",
            phase,
            visual_question,
            started_at_us,
            time.time_ns() // 1000,
            visual_data,
            resolved_data.model_copy(deep=True),
            dict(sources),
            tuple(observations),
            passages,
            omitted_count,
            issue,
        )
        if on_state is not None:
            await on_state(state)
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        return state

    async def resolve_fields(instructions, requested_fields, *, document_backed):
        """Validate individual field values and their source before publishing partial progress."""
        nonlocal resolved_data
        output_schema = {name: result_model.model_json_schema()["properties"][name] for name in requested_fields}
        # Keep the envelope explicit: the result model describes individual fields,
        # while values/sources are the generic protocol shared with the language model.
        response_properties = {
            "values": {"type": "object", "properties": output_schema, "additionalProperties": False},
        }
        if document_backed:
            response_properties["sources"] = {
                "type": "object",
                "properties": {
                    name: {
                        "type": "object",
                        "properties": {
                            "passage": {"type": "integer", "minimum": 0},
                            "quote": {"type": "string", "minLength": 1},
                        },
                        "required": ["passage", "quote"],
                        "additionalProperties": False,
                    }
                    for name in requested_fields
                },
                "additionalProperties": False,
            }
        response_schema = {
            "type": "object",
            "properties": response_properties,
            "required": list(response_properties),
            "additionalProperties": False,
            "$defs": result_model.model_json_schema().get("$defs", {}),
        }
        response = await llm.chat(
            (
                ChatMessage(role="system", content=instructions),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "visual_data": visual_data.model_dump(mode="json"),
                            "requested_fields": output_schema,
                            "passages": passages if document_backed else (),
                            "response_schema": response_schema,
                        }
                    ),
                ),
            ),
            max_tokens=512,
            temperature=0.0,
            enable_thinking=False,
            timeout=timeout_seconds,
        )
        if response.tool_calls:
            raise ValueError("field resolution cannot invoke model tools")
        response_data = json.loads(response.content)
        if not isinstance(response_data, dict):
            raise ValueError("field response must be an object")
        values = response_data.get("values")
        if not isinstance(values, dict) or set(values) - set(requested_fields):
            raise ValueError("field response contains unexpected fields")
        response_sources = response_data.get("sources", {})
        if not isinstance(response_sources, dict):
            raise ValueError("field sources must be a field-keyed object")
        resolved_values = resolved_data.model_dump()
        for name, value in values.items():
            if value is None:
                continue
            field_source = FieldSource("generated")
            if document_backed:
                source = response_sources.get(name, {})
                if not isinstance(source, dict):
                    continue
                passage_index, quote = source.get("passage"), source.get("quote")
                if (
                    type(passage_index) is not int
                    or not 0 <= passage_index < len(passages)
                    or not isinstance(quote, str)
                    or not quote.strip()
                    or quote not in passages[passage_index]["text"]
                ):
                    continue
                field_source = FieldSource("retrieved", passages[passage_index]["source"], quote)
            try:
                validated_value = TypeAdapter(result_model.model_fields[name].rebuild_annotation()).validate_python(
                    value
                )
            except ValidationError:
                # A malformed field must not discard other source-supported fields.
                continue
            resolved_values[name] = validated_value
            resolved_data = result_model.model_validate(resolved_values)
            sources[name] = field_source
            await report("extracting" if document_backed else "generating")

    while True:
        await report("observing")
        async with asyncio.timeout(timeout_seconds):
            frame = await frame_source()
        if frame is None:
            await report("observing", issue="A current camera frame is unavailable.")
            await asyncio.sleep(poll_interval_seconds)
            continue
        frame_position = (frame.timestamp_us, frame.sequence)
        if frame_position <= latest_frame_keys.get(frame.track_id, (-1, -1)):
            await asyncio.sleep(poll_interval_seconds)
            continue
        latest_frame_keys[frame.track_id] = frame_position
        async with asyncio.timeout(timeout_seconds):
            response = await observer.execute(
                ImageQueryRequest(
                    image=frame.image,
                    query=visual_question
                    + "\nReturn JSON matching this response schema: "
                    + json.dumps(visual_model.model_json_schema()),
                )
            )
        previous_visual_data = visual_data
        issue = ""
        try:
            visual_data = visual_model.model_validate_json(response.text) if response.available else None
        except ValidationError as error:
            visual_data = None
            issue = str(error)[:observation_text_limit]
        if visual_data is not None and previous_visual_data is not None and visual_data != previous_visual_data:
            resolved_data = result_model.model_validate({})
            sources.clear()
        sampled_at_us = time.time_ns() // 1000
        observations.append(
            VisionRAGRetrievalObservation(
                sampled_at_us,
                (frame.track_id, frame.sequence, frame.timestamp_us),
                response.text[:observation_text_limit],
                visual_data is not None,
                len(response.text) > observation_text_limit,
            )
        )
        while len(observations) > history_limit or (
            history_max_age_seconds is not None
            and observations
            and sampled_at_us - observations[0].timestamp_us > history_max_age_seconds * 1_000_000
        ):
            observations.popleft()
            omitted_count += 1
        if visual_data is None:
            await report("observing", issue=issue or "Visual data does not satisfy the requested schema yet.")
            await asyncio.sleep(poll_interval_seconds)
            continue

        await report("retrieving")
        async with asyncio.timeout(timeout_seconds):
            retrieved = await retrieval.execute(
                RetrieveRequest(
                    query=retrieval_query(visual_data),
                    top_k=max_retrieved_passages,
                )
            )
        passages = tuple(
            {
                "source": passage.source,
                "text": passage.text[:retrieved_text_limit],
                "truncated": len(passage.text) > retrieved_text_limit,
            }
            for passage in retrieved.results[:max_retrieved_passages]
        )
        missing_fields = [name for name, value in resolved_data.model_dump().items() if value is None]
        if passages:
            await report("extracting")
            await resolve_fields(extraction_instructions, missing_fields, document_backed=True)
        missing_fields = [name for name, value in resolved_data.model_dump().items() if value is None]
        if missing_fields:
            await report("generating")
            await resolve_fields(fallback_instructions, missing_fields, document_backed=False)
        if all(value is not None for value in resolved_data.model_dump().values()):
            return await report("complete", complete=True)
        # Usable visual data is not permission to invent unavailable results.
        # A new image may reveal better visual data; preserve validated results.
        await report("observing", issue="Requested fields remain unavailable.")
        await asyncio.sleep(poll_interval_seconds)
