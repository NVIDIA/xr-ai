# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/llm/nemotron3_nano`` (Nemotron-3-Nano-30B via vLLM).

vLLM's ``--reasoning-parser nano_v3`` writes reasoning into the
``reasoning`` response field.

Nemotron-3-Nano's chat template defaults to thinking-on, so callers
that pass ``enable_thinking=False`` would otherwise still get reasoning
output — typically burning the whole ``max_tokens`` budget on hidden
reasoning, returning ``finish_reason="length"`` with empty content and
no tool_calls. The ``default_extras`` below pins the wire-level default
to off; per-call ``enable_thinking=True`` still overrides via the
nested merge in ``openai_compat._build_payload``.
"""

NEMOTRON3_NANO = {
    "category":        "llm",
    "kind":            "openai_compat",
    "model_name":      "llm",
    "reasoning_field": "reasoning",
    "default_extras": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
        "reasoning":  True,
    },
}
