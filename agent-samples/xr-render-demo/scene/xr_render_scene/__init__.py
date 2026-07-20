# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native scene capability owned by the XR render demo."""

from .client import SceneClient
from .functions import (
    SceneControlFunctionsConfig,
    SceneObjectFunctionsConfig,
    SceneStateFunctionsConfig,
    SceneUpdateFunctionsConfig,
)
from .schemas import (
    AddPrimitiveRequest,
    EmptyRequest,
    RemovePrimitiveRequest,
    UpdatePrimitiveRequest,
)

__all__ = [
    "AddPrimitiveRequest",
    "EmptyRequest",
    "RemovePrimitiveRequest",
    "SceneClient",
    "SceneControlFunctionsConfig",
    "SceneObjectFunctionsConfig",
    "SceneStateFunctionsConfig",
    "SceneUpdateFunctionsConfig",
    "UpdatePrimitiveRequest",
]
