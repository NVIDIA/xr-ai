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
from ._gpu import (
    GPUDevice,
    GPUHardwareProfile,
    GPUInventoryError,
    GPUProcess,
    detect_gpu_config,
    load_gpu_hardware_profile,
    match_gpu_config,
    query_gpu_inventory,
)
from ._models import ModelDeployment, load_deployment_profile, load_model_deployment
from ._processes import ManagedProcess
from ._stack import Parallel, Process, run_stack
from ._vram import (
    VRAM_UTILIZATION_ENV,
    GPUPreflight,
    ServiceReservation,
    VRAMProfile,
    VRAMProfileError,
    derive_gpu_memory_utilization,
    format_vram_preflight,
    load_service_reservation,
    preflight_vram,
    read_service_port,
    require_vram_preflight,
    resolve_vram_profile,
    service_config_fingerprint,
    utilization_overrides,
)

__all__ = [
    "XR_RUNTIME_VAR", "load_cloudxr_env",
    "NATIVE_DEVICE_PROFILES", "is_native_profile", "read_device_profile",
    "ensure_credentials", "load_credentials", "require_credentials", "warn_if_missing",
    "read_config_scalar",
    "GPUDevice", "GPUHardwareProfile", "GPUInventoryError", "GPUProcess",
    "detect_gpu_config", "load_gpu_hardware_profile", "match_gpu_config",
    "query_gpu_inventory",
    "VRAM_UTILIZATION_ENV", "GPUPreflight", "ServiceReservation",
    "VRAMProfile", "VRAMProfileError", "derive_gpu_memory_utilization",
    "format_vram_preflight", "load_service_reservation", "preflight_vram",
    "read_service_port",
    "require_vram_preflight", "resolve_vram_profile",
    "service_config_fingerprint", "utilization_overrides",
    "ModelDeployment", "load_deployment_profile", "load_model_deployment",
    "ManagedProcess",
    "Parallel", "Process", "run_stack",
]
