"""
xr-ai-launcher — process management for the xr-ai stack.

Intentionally stdlib-only so it can be added to any sample without pulling
in the dependency chain of the processes it manages.

    from xr_ai_launcher import HubLauncher

    async with HubLauncher():
        await my_agent.run()
"""

from ._processes import ManagedProcess
from ._hub import HubLauncher

__all__ = ["ManagedProcess", "HubLauncher"]
