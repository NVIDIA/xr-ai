# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/llm/llama_nemotron`` (Llama-3.1-Nemotron-Nano-8B via vLLM)."""

LLAMA_NEMOTRON = {
    "category":   "llm",
    "kind":       "openai_compat",
    "model_name": "llm",
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
    },
}
