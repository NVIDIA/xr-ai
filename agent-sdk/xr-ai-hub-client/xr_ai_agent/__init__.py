# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the renamed :mod:`xr_ai_hub` package.

``xr_ai_agent`` (distribution ``xr-ai-agent``) was renamed to ``xr_ai_hub``
(distribution ``xr-ai-hub-client``). The public API is unchanged. Import from
``xr_ai_hub`` instead; this alias will be removed in a future version.
"""

import warnings

import xr_ai_hub as _xr_ai_hub
from xr_ai_hub import *  # noqa: F401,F403 -- re-export the public API

__all__ = list(_xr_ai_hub.__all__)

warnings.warn(
    "xr_ai_agent is deprecated; import from xr_ai_hub instead. "
    "This alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
