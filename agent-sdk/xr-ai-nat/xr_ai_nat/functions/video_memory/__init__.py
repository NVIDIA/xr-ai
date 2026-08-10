# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public video-memory functions and invocation schemas."""

from ._client import (
    HistoricalFrameRequest,
    HistoricalFrameResult,
    ListRecordedParticipantsRequest,
    ListRecordedParticipantsResult,
    QueryVideoRequest,
    QueryVideoResult,
    VideoHealthRequest,
    VideoHealthResult,
    VideoStatsRequest,
    VideoStatsResult,
)
from .functions import VideoMemoryFunctionsConfig

# Deprecated alias for the pre-rename public name (same data contract). Kept so
# `from xr_ai_nat.functions.video_memory import ParticipantsResult` keeps working.
ParticipantsResult = ListRecordedParticipantsResult

__all__ = [
    "HistoricalFrameRequest",
    "HistoricalFrameResult",
    "ListRecordedParticipantsRequest",
    "ListRecordedParticipantsResult",
    "ParticipantsResult",
    "QueryVideoRequest",
    "QueryVideoResult",
    "VideoMemoryFunctionsConfig",
    "VideoHealthRequest",
    "VideoHealthResult",
    "VideoStatsRequest",
    "VideoStatsResult",
]
