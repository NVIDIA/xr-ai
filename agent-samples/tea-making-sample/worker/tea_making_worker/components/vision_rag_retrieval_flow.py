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


class _ResolutionResponseError(ValueError):
    """A malformed model envelope can retry using already retrieved passages."""


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
            Field validators receive source_kind and exact source_quote in Pydantic
            validation context, so application quantities can enforce evidence constraints.
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
    retrieved_for: VisualData | None = None
    resolved_for: VisualData | None = None
    field_evidence: dict[str, FieldSource] = {}
    resolution_issues: dict[str, str] = {}
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
            "; ".join(filter(None, (issue, *(f"{name}: {reason}" for name, reason in resolution_issues.items()))))[
                :observation_text_limit
            ],
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
                            "passage_id": {
                                "type": "string",
                                "enum": [passage["passage_id"] for passage in passages],
                                "description": "Copy an exact passage_id from the supplied passages.",
                            },
                            "quote": {"type": "string", "minLength": 1},
                        },
                        "required": ["passage_id", "quote"],
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
                            "passages": passages,
                            "field_evidence": {
                                name: {"source": evidence.source, "quote": evidence.quote}
                                for name, evidence in field_evidence.items()
                                if name in requested_fields
                            },
                            "validation_feedback": dict(resolution_issues),
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
            raise _ResolutionResponseError("Field resolution cannot invoke model tools.")
        try:
            response_data = json.loads(response.content)
        except json.JSONDecodeError as error:
            raise _ResolutionResponseError("Field response is not valid JSON.") from error
        if not isinstance(response_data, dict):
            raise _ResolutionResponseError("Field response must be an object.")
        values = response_data.get("values")
        if not isinstance(values, dict) or set(values) - set(requested_fields):
            raise _ResolutionResponseError("Field response contains unexpected fields.")
        response_sources = response_data.get("sources", {})
        if not isinstance(response_sources, dict):
            raise _ResolutionResponseError("Field sources must be a field-keyed object.")
        resolution_issues.pop("response", None)
        resolved_values = resolved_data.model_dump()
        passage_lookup = {passage["passage_id"]: passage for passage in passages}
        valid_response = True
        for name in requested_fields:
            value = values.get(name)
            if document_backed:
                source = response_sources.get(name)
                if value is None and source is None:
                    continue
                passage_id = source.get("passage_id") if isinstance(source, dict) else None
                if not isinstance(passage_id, str) or passage_id not in passage_lookup:
                    resolution_issues[name] = "The source must cite an exact supplied passage_id."
                    valid_response = False
                    continue
                passage = passage_lookup[passage_id]
                quote = source.get("quote")
                if not isinstance(quote, str) or not quote.strip() or quote not in passage["text"]:
                    resolution_issues[name] = "The quote is not present in the cited passage."
                    valid_response = False
                    continue
                # A valid quote remains useful evidence even when it supplies a
                # range rather than a directly usable scalar result.
                field_evidence[name] = FieldSource("retrieved", passage["source"], quote)
            if value is None:
                continue
            evidence = field_evidence.get(name)
            source_kind = "retrieved" if document_backed else "generated"
            try:
                validated_value = TypeAdapter(result_model.model_fields[name].rebuild_annotation()).validate_python(
                    value,
                    context={"source_kind": source_kind, "source_quote": evidence.quote if evidence else ""},
                )
            except ValidationError as error:
                resolution_issues[name] = error.errors()[0]["msg"]
                valid_response = False
                continue
            resolved_values[name] = validated_value
            resolved_data = result_model.model_validate(resolved_values)
            sources[name] = evidence if document_backed else FieldSource("generated")
            resolution_issues.pop(name, None)
            await report("extracting" if document_backed else "generating")
        return valid_response

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
        issue = ""
        try:
            visual_data = visual_model.model_validate_json(response.text) if response.available else None
        except ValidationError as error:
            visual_data = None
            issue = str(error)[:observation_text_limit]
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

        if visual_data != retrieved_for:
            # Keep only one visual input's evidence. A new entity must never
            # inherit partial results or generation constraints from the old one.
            resolved_data = result_model.model_validate({})
            sources.clear()
            field_evidence.clear()
            resolution_issues.clear()
            resolved_for = None
            await report("retrieving")
            async with asyncio.timeout(timeout_seconds):
                retrieved = await retrieval.execute(
                    RetrieveRequest(query=retrieval_query(visual_data), top_k=max_retrieved_passages)
                )
            passages = tuple(
                {
                    "passage_id": f"P{index + 1}",
                    "source": passage.source,
                    "text": passage.text[:retrieved_text_limit],
                    "truncated": len(passage.text) > retrieved_text_limit,
                }
                for index, passage in enumerate(retrieved.results[:max_retrieved_passages])
            )
            # Cache only a completed retrieval, not a failed operation.
            retrieved_for = visual_data.model_copy(deep=True)

        if visual_data == resolved_for:
            await report("observing", issue="Requested fields remain unavailable; waiting for new visual information.")
            await asyncio.sleep(poll_interval_seconds)
            continue

        extraction_complete = not passages
        while True:
            missing_fields = [name for name, value in resolved_data.model_dump().items() if value is None]
            try:
                if not extraction_complete and missing_fields:
                    await report("extracting")
                    extraction_complete = await resolve_fields(
                        extraction_instructions, missing_fields, document_backed=True
                    )
                missing_fields = [name for name, value in resolved_data.model_dump().items() if value is None]
                valid_generation = True
                if missing_fields:
                    await report("generating")
                    valid_generation = await resolve_fields(
                        fallback_instructions, missing_fields, document_backed=False
                    )
            except _ResolutionResponseError as error:
                # Repair malformed output against the already known input. Do
                # not reacquire/re-read frames or retrieve the same documents.
                resolution_issues["response"] = str(error)
                await report("extracting")
                await asyncio.sleep(poll_interval_seconds)
                continue
            if all(value is not None for value in resolved_data.model_dump().values()):
                return await report("complete", complete=True)
            if extraction_complete and valid_generation:
                # A validated partial/null outcome needs new visual information,
                # rather than another identical extraction/generation request.
                resolved_for = visual_data.model_copy(deep=True)
                break
            await report("generating")
            await asyncio.sleep(poll_interval_seconds)
        await report("observing", issue="Requested fields remain unavailable.")
        await asyncio.sleep(poll_interval_seconds)
