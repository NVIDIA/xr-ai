# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the former ``text_memory.schemas`` module.

The schema classes folded into :mod:`xr_ai_nat.functions.text_memory.functions`
when text-memory adopted a typed request/result API. Import surviving names from
:mod:`xr_ai_nat.functions.text_memory` instead of this module; this alias will be
removed in a future version.

``TranscriptSegment`` is unchanged. ``TranscriptStats`` was renamed to
``TranscriptStatsResult`` but keeps the same fields (only ``earliest_us`` /
``latest_us`` widened ``int`` → ``int | None``), so it remains here as a
deprecated alias. ``OperationResult`` and ``TextMemoryError`` were genuinely
removed (the typed API validates input and returns typed results instead of
error-as-data) and are not aliased.
"""

import warnings

from .functions import TranscriptSegment, TranscriptStatsResult

# Deprecated alias — same fields as the old TranscriptStats (bounds widened).
TranscriptStats = TranscriptStatsResult

warnings.warn(
    "xr_ai_nat.functions.text_memory.schemas is deprecated; import from "
    "xr_ai_nat.functions.text_memory instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TranscriptSegment", "TranscriptStats"]
