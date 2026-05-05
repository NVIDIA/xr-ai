# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-ai-launcher — process management for the xr-ai stack.

Intentionally stdlib-only so it can be added to any sample without pulling
in the dependency chain of the processes it manages.

Typical usage::

    from xr_ai_launcher import Parallel, Process, run_stack

    _BASE = Path(__file__).resolve().parent

    PROCESSES = [
        Process("hub",    "../../server-runtime", "xr_media_hub"),
        Parallel([
            Process("stt", "../../ai-services/stt-server", "stt_server"),
            Process("tts", "../../ai-services/tts/piper",  "piper_tts_server"),
        ]),
        Process("worker", "worker", "my_agent_worker"),
    ]

    def run() -> None:
        run_stack(PROCESSES, _BASE)
"""

from ._cloudxr_env import XR_RUNTIME_VAR, load_cloudxr_env
from ._credentials import ensure_credentials, load_credentials
from ._processes import ManagedProcess
from ._stack import Parallel, Process, run_stack

__all__ = [
    "XR_RUNTIME_VAR", "load_cloudxr_env",
    "ensure_credentials", "load_credentials",
    "ManagedProcess",
    "Parallel", "Process", "run_stack",
]
