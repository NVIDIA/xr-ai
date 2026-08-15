# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``services/nemotron-omni-llm`` (Nemotron-3-Nano-Omni-30B via vLLM).

Multimodal text + video.  vLLM's ``--reasoning-parser nemotron_v3`` writes
reasoning into the ``reasoning_content`` response field.

The model's chat template defaults to thinking-on.  Keep ordinary calls
non-reasoning by default so short responses do not spend their token budget on
hidden reasoning; callers can still opt in with ``enable_thinking=True``.
"""

NEMOTRON_OMNI = {
    "category":        "llm",
    "kind":            "openai_compat",
    "model_name":      "llm",
    "reasoning_field": "reasoning_content",
    "default_extras": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
        "vision":     True,
        "video":      True,
        "reasoning":  True,
    },
}
