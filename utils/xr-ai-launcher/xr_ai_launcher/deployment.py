# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment readiness hook for hosts running an existing sample command.

Set ``XR_AI_STACK_READY_FILE`` to a private writable path. ``run_stack`` writes
the current ``XR_AI_LOG_TIMESTAMP`` there once all owned processes are ready,
and removes the file on shutdown. The supervising host must also check process
liveness; the file can remain after SIGKILL or a machine crash. Model convenience
stacks using ``exit_after_ready=True`` do not publish this signal.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["stack_readiness"]


@contextmanager
def stack_readiness(enabled: bool = True) -> Iterator[Callable[[], None]]:
    """Provide a readiness publisher scoped to one foreground stack lifetime."""
    raw = os.environ.get("XR_AI_STACK_READY_FILE") if enabled else None
    path = Path(raw) if raw else None
    if path:
        path.unlink(missing_ok=True)

    def ready() -> None:
        if path:
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            try:
                temporary.write_text(os.environ.get("XR_AI_LOG_TIMESTAMP", "ready") + "\n")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    try:
        yield ready
    finally:
        if path:
            path.unlink(missing_ok=True)
