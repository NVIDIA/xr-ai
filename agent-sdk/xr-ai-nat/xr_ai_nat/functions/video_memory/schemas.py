# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for video-memory contracts.

The typed request/result models moved into
:mod:`xr_ai_nat.functions.video_memory._client`. This module re-exports the
models whose names are unchanged so existing imports keep working; import from
:mod:`xr_ai_nat.functions.video_memory` or its ``._client`` submodule instead.

Some models were renamed in the same move and are intentionally NOT re-exported
here (breaking change): ``EmptyRequest`` split into ``VideoHealthRequest`` /
``ListRecordedParticipantsRequest``, ``ParticipantsResult`` became
``ListRecordedParticipantsResult``, and ``VideoMemoryHealth`` became
``VideoHealthResult``.
"""

import warnings

from ._client import (
    HistoricalFrameRequest,
    HistoricalFrameResult,
    QueryVideoRequest,
    QueryVideoResult,
    VideoStatsRequest,
    VideoStatsResult,
)

warnings.warn(
    "xr_ai_nat.functions.video_memory.schemas is deprecated; import these models "
    "from xr_ai_nat.functions.video_memory or its ._client submodule instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HistoricalFrameRequest",
    "HistoricalFrameResult",
    "QueryVideoRequest",
    "QueryVideoResult",
    "VideoStatsRequest",
    "VideoStatsResult",
]
