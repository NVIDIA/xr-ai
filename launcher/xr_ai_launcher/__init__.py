# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-ai-launcher — process management for the xr-ai stack.

Intentionally stdlib-only so it can be added to any sample without pulling
in the dependency chain of the processes it manages.

Typical usage — thin orchestrator backed by a stack.toml::

    from xr_ai_launcher import run_stack

    def run() -> None:
        asyncio.run(run_stack())

Advanced usage — compose with custom async logic::

    from xr_ai_launcher import StackLauncher

    async with StackLauncher("stack.toml") as procs:
        await my_loop()
"""

from ._cloudxr_env import (
    XR_RUNTIME_VAR,
    load_cloudxr_env,
    wait_for_cloudxr_env,
    wait_for_cloudxr_runtime_started,
)
from ._credentials import ensure_credentials, load_credentials
from ._processes import ManagedProcess
from ._project import ProjectLauncher
from ._hub import HubLauncher
from ._stack import Process, StackLauncher, run_stack

__all__ = [
    "XR_RUNTIME_VAR",
    "ensure_credentials", "load_credentials",
    "load_cloudxr_env", "wait_for_cloudxr_env", "wait_for_cloudxr_runtime_started",
    "ManagedProcess", "ProjectLauncher", "HubLauncher",
    "Process", "StackLauncher", "run_stack",
]
