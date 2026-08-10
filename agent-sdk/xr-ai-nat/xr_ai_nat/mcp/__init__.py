# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose selected native NAT functions to MCP-only agents."""

from .server import create_mcp_server

__all__ = ["create_mcp_server"]
