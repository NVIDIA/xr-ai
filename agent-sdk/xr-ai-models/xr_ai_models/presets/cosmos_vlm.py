# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preset for ``ai-services/vlm-server`` (Cosmos-Reason1-7B via vLLM).

Cosmos-Reason1-7B is a Qwen2.5-VL-7B fine-tune that accepts image *and* video
content blocks.  Video is opt-in at the server: vlm-server's
``max_videos_per_prompt`` defaults to 0, since vLLM reserves tens of GiB of
activation memory at startup once it is set non-zero.  Callers that send
``video_url`` content must set it to >= 1 in the server YAML.
"""

COSMOS_VLM = {
    "category":   "vlm",
    "kind":       "openai_compat",
    "model_name": "vlm",
    "capabilities": {
        "streaming": True,
        "vision":    True,
        "video":     True,
    },
    "default_extras": {
        # Cosmos-Reason emits <think>…</think> blobs by default; turn that off
        # so VLM content is the plain answer the worker can show / speak.
        "chat_template_kwargs": {"enable_thinking": False},
    },
}
