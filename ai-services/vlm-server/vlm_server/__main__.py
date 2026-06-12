# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vlm_server — vLLM launcher for Cosmos-Reason1-7B (or any Qwen2.5-VL-compatible VLM).

Reads config, builds vLLM serve flags, and dispatches through ``xr_ai_vllm.serve``
to either the pip-installed ``vllm`` CLI or the NGC ``nvcr.io/nvidia/vllm``
docker container — picked per-YAML via ``vllm_backend: pip|docker``.

Serves vLLM's OpenAI-compatible /v1/chat/completions endpoint; images are
passed as base64 data URLs in the ``image_url`` content block.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:                   str    HuggingFace model ID.
    host:                    str    Bind address (default: "0.0.0.0").
    port:                    int    HTTP port (default: 8100).
    served_model_name:       str    Name exposed in /v1/models (default: "vlm").
    hf_token:                str    HuggingFace token for gated models.
    model_cache:             str    HF weight cache, relative to this YAML.
    max_num_seqs:            int    vLLM --max-num-seqs (default: 4).
    tensor_parallel_size:    int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:           int    vLLM --max-model-len (default: 8192).
    gpu_memory_utilization:  float  vLLM --gpu-memory-utilization (default: 0.85).
    enforce_eager:           bool   Skip CUDA graph capture (default: false).
    max_images_per_prompt:   int    Max images per request (default: 1).
    max_videos_per_prompt:   int    Max video items per request (default: 0).
                                    Set >0 only if your worker sends video;
                                    0 skips vLLM's video activation profiling
                                    at startup, saving tens of GiB on
                                    Qwen2.5-VL-class models.
    vllm_backend:            str    "pip" (default) or "docker".
    vllm_image:              str    NGC image when vllm_backend=docker
                                    (default: nvcr.io/nvidia/vllm:26.04-py3).
"""
import json
import os
import sys

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_vllm import (
    DEFAULT_IMAGE,
    load_config,
    resolve_model_cache,
    serve,
    setup_hf_env,
)

_DEFAULT_PORT        = 8100
_DEFAULT_HOST        = "0.0.0.0"
_DEFAULT_SERVED_NAME = "vlm"
_DEFAULT_SEQS        = 4
_DEFAULT_TP          = 1
_DEFAULT_CTX         = 8192
_DEFAULT_GPU_MEM     = 0.85
_DEFAULT_EAGER       = False
_DEFAULT_MAX_IMAGES  = 1
_DEFAULT_MAX_VIDEOS  = 0

_CONTAINER_NAME = "xr-ai-vllm-vlm-server"


def run() -> None:
    setup_logging("vlm")

    cfg, yaml_dir, ready_file = load_config()

    if not cfg.get("model"):
        logger.error("'model' is required in config")
        sys.exit(1)

    model         = cfg["model"]
    host          = cfg.get("host",                 _DEFAULT_HOST)
    port          = int(cfg.get("port",             _DEFAULT_PORT))
    served_name   = cfg.get("served_model_name",    _DEFAULT_SERVED_NAME)
    max_seqs      = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size       = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx       = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem       = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = bool(cfg.get("enforce_eager",   _DEFAULT_EAGER))
    max_images    = int(cfg.get("max_images_per_prompt", _DEFAULT_MAX_IMAGES))
    max_videos    = int(cfg.get("max_videos_per_prompt", _DEFAULT_MAX_VIDEOS))
    backend       = cfg.get("vllm_backend",         "pip")
    image         = cfg.get("vllm_image",           DEFAULT_IMAGE)

    model_cache = resolve_model_cache(cfg, yaml_dir, default="../models")
    cuda_devices = setup_hf_env(cfg, model_cache)

    extra_serve_args = [
        "--served-model-name", served_name,
        "--trust-remote-code",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", str(gpu_mem),
        "--limit-mm-per-prompt", json.dumps({"image": max_images, "video": max_videos}),
    ]
    if enforce_eager:
        extra_serve_args.append("--enforce-eager")

    serve(
        backend=backend,
        persistent=True,
        image=image,
        container_name=_CONTAINER_NAME,
        log_prefix="vlm_server",
        model=model,
        extra_serve_args=extra_serve_args,
        host=host,
        port=port,
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN") or None,
        cuda_visible_devices=cuda_devices,
        ready_file=ready_file,
    )


if __name__ == "__main__":
    run()
