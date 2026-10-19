# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight CloudXR configuration shared by services and composition code."""

from ._env import (
    NATIVE_DEVICE_PROFILES,
    XR_RUNTIME_VAR,
    is_native_profile,
    load_cloudxr_env,
    read_device_profile,
)

__all__ = [
    "NATIVE_DEVICE_PROFILES",
    "XR_RUNTIME_VAR",
    "is_native_profile",
    "load_cloudxr_env",
    "read_device_profile",
]
