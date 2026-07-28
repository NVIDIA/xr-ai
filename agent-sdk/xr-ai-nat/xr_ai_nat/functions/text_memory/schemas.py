# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the former ``text_memory.schemas`` module.

The schema classes folded into :mod:`xr_ai_nat.functions.text_memory.functions`
when text-memory adopted a typed request/result API. Only ``TranscriptSegment``
survived; ``OperationResult`` and ``TextMemoryError`` were removed (the typed
API validates input and returns typed results instead of error-as-data), and
``TranscriptStats`` was renamed to ``TranscriptStatsResult``. Import surviving
names from :mod:`xr_ai_nat.functions.text_memory` instead of this module.
"""

import warnings

from .functions import TranscriptSegment

warnings.warn(
    "xr_ai_nat.functions.text_memory.schemas is deprecated; import from "
    "xr_ai_nat.functions.text_memory instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TranscriptSegment"]
