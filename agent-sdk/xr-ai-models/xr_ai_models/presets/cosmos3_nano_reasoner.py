# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for the Cosmos3 Nano Reasoner served by ``vlm-server``.

The local server uses standard ``vllm serve nvidia/Cosmos3-Nano``, which loads
the checkpoint's Reasoner. The separate Generator path requires vLLM's
``--omni`` mode and is intentionally not used by xr-ai.

Video requests require ``max_videos_per_prompt >= 1`` in vlm-server's YAML;
the server keeps video disabled by default to avoid reserving unused
multimodal activation memory.
"""

COSMOS3_NANO_REASONER = {
    "category":   "vlm",
    "kind":       "openai_compat",
    "model_name": "vlm",
    "capabilities": {
        "streaming": True,
        "vision":    True,
        "video":     True,
    },
}
