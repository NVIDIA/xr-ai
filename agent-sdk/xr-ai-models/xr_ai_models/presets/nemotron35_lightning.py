# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for the text-only Nemotron 3.5 Lightning vLLM service.

The ``nemotron_v3`` parser returns hidden reasoning through ``reasoning``.
Thinking is disabled by default so short agent calls
retain their answer-token budget; callers can opt in explicitly.
"""

NEMOTRON35_LIGHTNING = {
    "category": "llm",
    "kind": "openai_compat",
    "model_name": "llm",
    "reasoning_field": "reasoning",
    "default_extras": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "capabilities": {
        "streaming": True,
        "tool_calls": True,
        "reasoning": True,
    },
}
