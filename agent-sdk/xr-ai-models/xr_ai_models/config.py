# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the models configuration surface.

The models-config models and loaders moved to the private
:mod:`xr_ai_models._config`. Import the public names from the package root
(``from xr_ai_models import load_models_config``) instead; this alias will be
removed in a future version. Names not re-exported at the package root —
``KIND_OPENAI_COMPAT``, ``ModelKind``, ``Category``, and ``Spec`` — remain
importable from here for existing callers.
"""

from __future__ import annotations

import warnings

from ._config import (
    KIND_OPENAI_COMPAT,
    AdapterSpec,
    Category,
    EmbeddingSpec,
    DeploymentSpec,
    EndpointSpec,
    LLMSpec,
    ModelKind,
    ModelsConfig,
    Spec,
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
    "AdapterSpec",
    "KIND_OPENAI_COMPAT",
    "Category",
    "EmbeddingSpec",
    "DeploymentSpec",
    "EndpointSpec",
    "LLMSpec",
    "ModelKind",
    "ModelsConfig",
    "STTSpec",
    "Spec",
    "TTSSpec",
    "VLMSpec",
    "load_models_config",
    "load_models_config_from_dict",
]
