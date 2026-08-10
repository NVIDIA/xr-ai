# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt native NAT functions and transcript storage to XR voice sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from typing import Any, TypeVar

from nat.plugin_api import Function
from pydantic import BaseModel
from xr_ai_voice import VoiceHandler, VoiceQuery, VoiceResponse, VoiceTurn

from ..events import EventDispatcher, EventTopic
from ..functions.text_memory import AddTranscriptRequest

_RequestMapper = Callable[[VoiceQuery], Any]
_ResponseMapper = Callable[[Any], str]
_EventPayloadT = TypeVar("_EventPayloadT", bound=BaseModel)
_EventResponseMapper = Callable[[tuple[Any, ...]], VoiceResponse]


def as_voice_handler(
    function: Function,
    *,
    request: _RequestMapper,
    response: _ResponseMapper,
    streaming: bool = False,
) -> VoiceHandler:
    """Map voice turns to one native NAT function and its textual response."""

    async def handle(query: VoiceQuery) -> VoiceResponse:
        function_request = request(query)
        if not streaming:
            return response(await function.ainvoke(function_request))

        async def stream() -> AsyncIterator[str]:
            async for chunk in function.astream(function_request):  # pyright: ignore[reportGeneralTypeIssues]
                if text := response(chunk):
                    yield text

        return stream()

    return handle


def as_voice_event_handler(
    dispatcher: EventDispatcher,
    topic: EventTopic[_EventPayloadT],
    *,
    payload: Callable[[VoiceQuery], _EventPayloadT | dict[str, Any]],
    producer: str = "voice.input",
    subscribers: Collection[str] | None = None,
    response: _EventResponseMapper | None = None,
) -> VoiceHandler:
    """Publish accepted voice turns as typed events and return a subscriber response."""

    def first_text(results: tuple[Any, ...]) -> str:
        return next((item for item in results if isinstance(item, str)), "")

    map_response = response or first_text

    async def handle(query: VoiceQuery) -> VoiceResponse:
        results = await dispatcher.publish(
            topic,
            participant_id=query.participant_id,
            producer=producer,
            payload=payload(query),
            subscribers=subscribers,
            timestamp_us=query.timestamp_us,
        )
        return map_response(results)

    return handle


def record_voice_transcripts(
    add_transcript: Function,
) -> Callable[[VoiceTurn], Awaitable[None]]:
    """Return a voice-session observer that stores completed user and agent turns."""

    async def record(turn: VoiceTurn) -> None:
        if turn.text.strip():
            await add_transcript.ainvoke(
                AddTranscriptRequest(
                    source_id=f"{turn.participant_id}:{turn.role}",
                    timestamp_us=turn.timestamp_us,
                    text=turn.text,
                )
            )

    return record


__all__ = ["as_voice_event_handler", "as_voice_handler", "record_voice_transcripts"]
