# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the models configuration surface.

The models-config models and loaders moved to the private
:mod:`xr_ai_models._config`. Import the public names from the package root
(``from xr_ai_models import load_models_config``) instead; this alias will be
removed in a future version. ``KIND_OPENAI_COMPAT`` / ``ModelKind`` remain
re-exported here for existing callers.
"""

from __future__ import annotations

import warnings

from ._config import (
    KIND_OPENAI_COMPAT,
    LLMSpec,
    ModelKind,
    ModelsConfig,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
    load_models_config_from_dict,
)

warnings.warn(
    "xr_ai_models.config is deprecated; import from xr_ai_models instead. "
    "This alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "KIND_OPENAI_COMPAT",
    "LLMSpec",
    "ModelKind",
    "ModelsConfig",
    "STTSpec",
    "TTSSpec",
    "VLMSpec",
    "load_models_config",
    "load_models_config_from_dict",
]
