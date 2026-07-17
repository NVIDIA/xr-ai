# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register XR AI function groups with NeMo Agent Toolkit."""

from .functions.spatial_math.functions import spatial_math_functions as _spatial_math_functions  # noqa: F401

__all__: list[str] = []
