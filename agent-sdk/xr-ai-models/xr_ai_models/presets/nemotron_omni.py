# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/llm/nemotron_omni`` (Nemotron-3-Nano-Omni-30B via vLLM).

Multimodal text + video.  vLLM's ``--reasoning-parser nemotron_v3`` writes
reasoning into the ``reasoning_content`` response field.
"""

NEMOTRON_OMNI = {
    "category":        "llm",
    "kind":            "openai_compat",
    "model_name":      "llm",
    "reasoning_field": "reasoning_content",
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
        "vision":     True,
        "video":      True,
        "reasoning":  True,
    },
}
