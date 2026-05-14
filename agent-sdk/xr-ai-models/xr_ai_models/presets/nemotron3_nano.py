# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/llm/nemotron3_nano`` (Nemotron-3-Nano-30B via vLLM).

vLLM's ``--reasoning-parser nano_v3`` writes reasoning into the
``reasoning`` response field.
"""

NEMOTRON3_NANO = {
    "category":        "llm",
    "kind":            "openai_compat",
    "model_name":      "llm",
    "reasoning_field": "reasoning",
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
        "reasoning":  True,
    },
}
