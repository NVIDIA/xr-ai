# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the shared coordinate value models.

`Vector3` and `SpatialFrame` moved to the capability-neutral
:mod:`xr_ai_nat.functions.types`. This module remains as a thin, deprecated
alias so existing callers that import ``xr_ai_nat.functions.spatial_math.schemas``
keep working. Import from ``xr_ai_nat.functions.types`` instead; this alias will
be removed in a future version.
"""

from __future__ import annotations

import warnings

from ..types import SpatialFrame, Vector3

warnings.warn(
    "xr_ai_nat.functions.spatial_math.schemas is deprecated; import SpatialFrame "
    "and Vector3 from xr_ai_nat.functions.types instead. This alias will be "
    "removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SpatialFrame", "Vector3"]
