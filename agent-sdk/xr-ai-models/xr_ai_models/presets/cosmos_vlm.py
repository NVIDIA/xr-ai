# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/vlm-server`` (Cosmos-Reason1-7B via vLLM)."""

COSMOS_VLM = {
    "category":   "vlm",
    "kind":       "openai_compat",
    "model_name": "vlm",
    "capabilities": {
        "streaming": True,
        "vision":    True,
    },
    "default_extras": {
        # Cosmos-Reason emits <think>…</think> blobs by default; turn that off
        # so VLM content is the plain answer the worker can show / speak.
        "chat_template_kwargs": {"enable_thinking": False},
    },
}
