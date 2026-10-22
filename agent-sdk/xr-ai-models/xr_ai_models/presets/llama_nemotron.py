# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``services/llama-nemotron-llm`` (Llama-3.1-Nemotron-Nano-8B via vLLM).

The model switches reasoning with a ``detailed thinking on|off`` system line
and reasons when neither is present.  The server does not strip the
reasoning preamble, so the endpoint default is off.
"""

LLAMA_NEMOTRON = {
    "category":   "llm",
    "kind":       "openai_compat",
    "model_name": "llm",
    "thinking":   {"control": "system_prompt", "default": False},
    "capabilities": {
        "streaming":  True,
        "tool_calls": True,
    },
}
