# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort terminal summaries for launcher-managed process failures."""
from __future__ import annotations

import shlex
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from ._processes import _configured_log_dir


@dataclass(frozen=True)
class FailureContext:
    """Metadata for one command constructed by the launcher."""

    service: str
    command: tuple[str, ...]
    project_path: Path
    config_path: Path | None


def emit_failure_summary(
    context: FailureContext,
    phase: str,
    *,
    returncode: int | None = None,
    error: BaseException | None = None,
) -> None:
    """Print the observed process failure and an available run-log directory."""
    lines = [
        "─" * 78,
        f"Process failure: {context.service} ({phase})",
        f"Command: {shlex.join(context.command)}",
        f"Project: {context.project_path}",
    ]
    if context.config_path is not None:
        lines.append(f"Config: {context.config_path}")

    if error is not None:
        lines.append(f"Error: {type(error).__name__}: {error}")
    elif returncode is not None and returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = "UNKNOWN"
        lines.append(f"Terminated by signal: {name} ({number})")
    else:
        lines.append(f"Exit code: {returncode}")

    log_dir = _configured_log_dir()
    if log_dir is not None and log_dir.is_dir():
        lines.append(f"Run logs: {log_dir}")
    else:
        lines.append("Run logs unavailable; review the terminal output.")
    lines.extend(("─" * 78, ""))
    print("\n".join(lines), file=sys.stderr, flush=True)


def failure_exit_status(returncode: int | None) -> int:
    """Return a nonzero shell status, mapping zero to 1 and signals to 128+N."""
    if returncode is None or returncode == 0:
        return 1
    if returncode < 0:
        return 128 + min(-returncode, 127)
    return returncode
