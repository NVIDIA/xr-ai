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
from .functions import VideoMemoryControlFunctionsConfig, VideoMemoryFunctionsConfig

__all__ = [
    "HistoricalFrameRequest",
    "HistoricalFrameResult",
    "ListRecordedParticipantsRequest",
    "ListRecordedParticipantsResult",
    "QueryVideoRequest",
    "QueryVideoResult",
    "VideoMemoryFunctionsConfig",
    "VideoMemoryControlFunctionsConfig",
    "VideoHealthRequest",
    "VideoHealthResult",
    "VideoStatsRequest",
    "VideoStatsResult",
]
