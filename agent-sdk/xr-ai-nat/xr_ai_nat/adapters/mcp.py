# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the MCP publisher.

The generic native-function → MCP publisher moved to :mod:`xr_ai_nat.mcp`.
This module remains as a thin, deprecated alias so existing callers that import
``xr_ai_nat.adapters.mcp.create_mcp_server`` keep working. Import from
``xr_ai_nat.mcp`` instead; this alias will be removed in a future version.
"""

from __future__ import annotations

import warnings

from xr_ai_nat.mcp.server import create_mcp_server

warnings.warn(
    "xr_ai_nat.adapters.mcp is deprecated; import create_mcp_server from "
    "xr_ai_nat.mcp instead. This alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["create_mcp_server"]
