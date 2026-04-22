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

from ._processes import ManagedProcess
from ._project import ProjectLauncher
from ._hub import HubLauncher
from ._stack import Process, StackLauncher, run_stack

__all__ = ["ManagedProcess", "ProjectLauncher", "HubLauncher", "Process", "StackLauncher", "run_stack"]
