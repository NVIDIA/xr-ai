# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native scene capability owned by the XR render demo."""

from .client import SceneClient
from .schemas import (
    AddPrimitiveRequest,
    AddPrimitiveResult,
    EmptyRequest,
    MutationResult,
    RemovePrimitiveRequest,
    SceneObject,
    SceneState,
    UpdatePrimitiveRequest,
)
from .tools import SceneTools

__all__ = [
    "AddPrimitiveRequest",
    "AddPrimitiveResult",
    "EmptyRequest",
    "MutationResult",
    "RemovePrimitiveRequest",
    "SceneClient",
    "SceneObject",
    "SceneState",
    "SceneTools",
    "UpdatePrimitiveRequest",
]
