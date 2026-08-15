# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record final speech transcripts through the file-output agent."""

from __future__ import annotations

from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VOICE_TRANSCRIPT_TOPIC, VoiceTranscript

from .events import TRANSCRIPT_RECORD_TOPIC, TranscriptRecord


class TranscriptAgent(Agent):
    """Convert final pre-gate STT results into sample transcript records."""

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def record(self, transcript: VoiceTranscript, ctx: RuntimeContext) -> None:
        await ctx.publish(
            TRANSCRIPT_RECORD_TOPIC,
            TranscriptRecord(
                timestamp_us=transcript.timestamp_us,
                text=transcript.text,
            ),
        )


__all__ = ["TranscriptAgent"]
