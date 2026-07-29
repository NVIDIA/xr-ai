# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the service factory helpers.

The ``make_*`` factory helpers moved to the private :mod:`xr_ai_models._factory`.
Import them from the package root (``from xr_ai_models import make_vlm``) instead;
this alias will be removed in a future version.
"""

from __future__ import annotations

import warnings

from ._factory import make_llm, make_stt, make_tts, make_vlm

warnings.warn(
    "xr_ai_models.factory is deprecated; import from xr_ai_models instead. "
    "This alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["make_llm", "make_stt", "make_tts", "make_vlm"]
