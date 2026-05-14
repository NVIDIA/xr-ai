# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/stt-server`` (Parakeet-TDT-0.6B via NeMo)."""

PARAKEET_STT = {
    "category": "stt",
    "kind":     "openai_compat",
    "timeout":  30.0,
}
