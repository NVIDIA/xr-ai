# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt native NAT functions and transcript storage to XR voice sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from nat.plugin_api import Function
from xr_ai_voice import VoiceHandler, VoiceQuery, VoiceResponse, VoiceTurn

from ..functions.text_memory import AddTranscriptRequest

_RequestMapper = Callable[[VoiceQuery], Any]
_ResponseMapper = Callable[[Any], str]


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


__all__ = ["as_voice_handler", "record_voice_transcripts"]
