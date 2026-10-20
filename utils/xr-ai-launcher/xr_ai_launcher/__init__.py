# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-ai-launcher — process management for the xr-ai stack.

Uses the standard library and the lightweight xr-ai-gpu utility so samples
do not pull in the dependency chain of the processes they manage.

Typical usage::

    from xr_ai_launcher import Parallel, Process, run_stack

    _BASE = Path(__file__).resolve().parent

    PROCESSES = [
        Process("hub",    "../../services/device-io-hub", "device_io_hub"),
        Parallel([
            Process("stt", "../../services/stt-server", "stt_server"),
            Process("tts", "../../services/pocket-tts",  "pocket_tts_server"),
        ]),
        Process("worker", "worker", "my_agent_worker"),
    ]

    def run() -> None:
        run_stack(PROCESSES, _BASE)
"""

from xr_ai_gpu import GPUInventoryError

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
from ._models import ModelDeployment, load_deployment_profile, load_model_deployment
from ._processes import ManagedProcess
from ._stack import Parallel, Process, run_stack

__all__ = [
    "XR_RUNTIME_VAR", "load_cloudxr_env",
    "NATIVE_DEVICE_PROFILES", "is_native_profile", "read_device_profile",
    "ensure_credentials", "load_credentials", "require_credentials", "warn_if_missing",
    "read_config_scalar",
    "GPUInventoryError", "detect_gpu_config",
    "ModelDeployment", "load_deployment_profile", "load_model_deployment",
    "ManagedProcess",
    "Parallel", "Process", "run_stack",
]
