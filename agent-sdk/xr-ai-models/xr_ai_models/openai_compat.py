# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the OpenAI-compatible clients.

The concrete ``OpenAICompat*`` clients moved to the private
:mod:`xr_ai_models._openai_compat`. Import them from the package root
(``from xr_ai_models import OpenAICompatVLM``) instead; this alias will be
removed in a future version.
"""

from __future__ import annotations

import warnings

from ._openai_compat import (
    OpenAICompatEmbedding,
    OpenAICompatLLM,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)

warnings.warn(
    "xr_ai_models.openai_compat is deprecated; import from xr_ai_models instead. "
    "This alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "OpenAICompatEmbedding",
    "OpenAICompatLLM",
    "OpenAICompatSTT",
    "OpenAICompatTTS",
    "OpenAICompatVLM",
]
