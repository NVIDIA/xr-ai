# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for video-memory contracts.

The typed request/result models moved into
:mod:`xr_ai_nat.functions.video_memory._client`. Import from
:mod:`xr_ai_nat.functions.video_memory` or its ``._client`` submodule instead;
this module remains as a deprecated forwarding alias and will be removed in a
future version.

Some models were renamed in the move but kept the same data contracts, so the
old names remain here as deprecated aliases: ``ParticipantsResult`` →
``ListRecordedParticipantsResult``, ``VideoMemoryHealth`` → ``VideoHealthResult``,
and the legacy no-argument ``EmptyRequest`` → ``ListRecordedParticipantsRequest``
(both are field-less strict requests).
"""

import warnings

from ._client import (
    HistoricalFrameRequest,
    HistoricalFrameResult,
    ListRecordedParticipantsRequest,
    ListRecordedParticipantsResult,
    QueryVideoRequest,
    QueryVideoResult,
    VideoHealthResult,
    VideoStatsRequest,
    VideoStatsResult,
)

# Deprecated legacy names — same data contracts as their renamed replacements.
EmptyRequest = ListRecordedParticipantsRequest
ParticipantsResult = ListRecordedParticipantsResult
VideoMemoryHealth = VideoHealthResult

warnings.warn(
    "xr_ai_nat.functions.video_memory.schemas is deprecated; import these models "
    "from xr_ai_nat.functions.video_memory or its ._client submodule instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EmptyRequest",
    "HistoricalFrameRequest",
    "HistoricalFrameResult",
    "ParticipantsResult",
    "QueryVideoRequest",
    "QueryVideoResult",
    "VideoMemoryHealth",
    "VideoStatsRequest",
    "VideoStatsResult",
]
