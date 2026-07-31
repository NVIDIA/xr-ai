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
        Process("hub",    "../../services/xr-media-hub", "xr_media_hub"),
        Parallel([
            Process("stt", "../../services/stt-server", "stt_server"),
            Process("tts", "../../services/piper-tts",  "piper_tts_server"),
        ]),
        Process("worker", "worker", "my_agent_worker"),
    ]

    def run() -> None:
        run_stack(PROCESSES, _BASE)
"""

from ._cloudxr_env import (
    NATIVE_DEVICE_PROFILES,
    XR_RUNTIME_VAR,
    is_native_profile,
    load_cloudxr_env,
    read_device_profile,
)
from ._config import read_config_scalar
from ._credentials import (
    ensure_credentials,
    load_credentials,
    require_credentials,
    warn_if_missing,
)
from ._gpu import detect_gpu_config
from ._models import ModelDeployment, load_model_deployment
from ._processes import ManagedProcess
from ._stack import Parallel, Process, run_stack

__all__ = [
    "XR_RUNTIME_VAR", "load_cloudxr_env",
    "NATIVE_DEVICE_PROFILES", "is_native_profile", "read_device_profile",
    "ensure_credentials", "load_credentials", "require_credentials", "warn_if_missing",
    "read_config_scalar",
    "detect_gpu_config",
    "ModelDeployment", "load_model_deployment",
    "ManagedProcess",
    "Parallel", "Process", "run_stack",
]
