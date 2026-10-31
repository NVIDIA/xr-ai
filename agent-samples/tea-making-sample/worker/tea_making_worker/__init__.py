# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local SOP agent and workflow-loading API.

Create SOPAgent with a loaded Workflow and existing typed model/tool services.
Bind its runtime, register it for notices, and use it as an SDK router destination.
Stop the router before the agent and its services.
"""

from .agents.sop.agent import SOPAgent
from .sop import Workflow, load_workflow

__all__ = ["SOPAgent", "Workflow", "load_workflow"]
