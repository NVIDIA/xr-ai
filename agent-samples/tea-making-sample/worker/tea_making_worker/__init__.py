# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local SOP agent and workflow-loading API.

Create SOPAgent with a loaded Workflow and existing typed model/tool services.
Register the agent once, bind its runtime, and stop it before closing services.
"""

from .agent import SOPAgent
from .sop import Workflow, load_workflow

__all__ = ["SOPAgent", "Workflow", "load_workflow"]
