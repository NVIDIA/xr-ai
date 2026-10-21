# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``services/nemotron3-nano-llm`` (Nemotron-3-Nano-30B via vLLM).

vLLM's ``--reasoning-parser nano_v3`` writes reasoning into the
``reasoning`` response field.

Nemotron-3-Nano's chat template defaults to thinking-on, which typically
burns the whole ``max_tokens`` budget on hidden reasoning and returns
``finish_reason="length"`` with empty content and no tool_calls.  The
endpoint default is therefore off; a per-call ``enable_thinking`` overrides
it in either direction.
"""

NEMOTRON3_NANO = {
    "category":        "llm",
    "kind":            "openai_compat",
    "model_name":      "llm",
    "reasoning_field": "reasoning",
    "thinking": {"control": "chat_template", "default": False},
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
        "reasoning":  True,
    },
}
