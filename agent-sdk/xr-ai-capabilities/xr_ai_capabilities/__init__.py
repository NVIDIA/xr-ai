# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable, framework-agnostic agent capabilities for xr-ai samples.

Capabilities are self-contained features an agent brain can compose — they talk
to the hub through a ``ProcessorEndpoint`` and depend only on the core SDK
(``xr-ai-agent`` / ``xr-ai-models``), not on any voice/pipeline framework.

Building blocks:

* :class:`VisionModule` — live-camera VLM question answering
* :class:`AgentCapability` — ABC for brain-local tools
* :class:`MCPToolset` — pairs an MCP client with the tool names it owns
* :func:`route_tool` — finds the right toolset for a given tool name
* :func:`collect_tool_defs` — queries all toolsets and returns :class:`~xr_ai_models.ToolDef` objects
"""
from .capability import AgentCapability
from .pixels import encode_image, frame_to_pil
from .toolset import McpClientProtocol, MCPToolset, collect_tool_defs, route_tool
from .vision import VISION_TOOL_NAME, VisionModule, VisionUnavailable

__all__ = [
    # Capability interface
    "AgentCapability",
    # MCP toolset routing
    "MCPToolset",
    "McpClientProtocol",
    "collect_tool_defs",
    "route_tool",
    # Vision capability
    "VISION_TOOL_NAME",
    "VisionModule",
    "VisionUnavailable",
    # Image encoding helpers
    "encode_image",
    "frame_to_pil",
]
