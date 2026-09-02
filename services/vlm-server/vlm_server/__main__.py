# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vlm_server — vLLM launcher for Cosmos3 Nano Reasoner (or another vLLM VLM).

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
    kv_cache_memory_bytes:   int    Explicit vLLM KV-cache size in bytes. Mutually
                                    exclusive with gpu_memory_utilization (optional).
    enforce_eager:           bool   Skip CUDA graph capture (default: false).
    async_scheduling:        bool   Enable vLLM async scheduling (default: false).
    hf_overrides:            dict   Hugging Face config overrides passed as JSON.
    mm_encoder_tp_mode:      str    Multimodal encoder TP mode (optional).
    max_images_per_prompt:   int    Max images per request (default: 1).
    max_videos_per_prompt:   int    Max video items per request (default: 0).
                                    Set >0 only if your worker sends video;
                                    0 skips vLLM's video activation profiling
                                    at startup.
    vllm_backend:            str    "pip" (default) or "docker".
    vllm_image:              str    NGC image when vllm_backend=docker
                                    (default: nvcr.io/nvidia/vllm:26.07-py3).
"""
import json
import os
import sys

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_vllm import (
    load_config,
    resolve_model_cache,
    serve,
    setup_hf_env,
)
from xr_ai_vllm._config import parse_config_bool

_DEFAULT_PORT        = 8100
_DEFAULT_HOST        = "0.0.0.0"
_DEFAULT_SERVED_NAME = "vlm"
_DEFAULT_SEQS        = 4
_DEFAULT_TP          = 1
_DEFAULT_CTX         = 8192
_DEFAULT_GPU_MEM     = 0.85
_DEFAULT_EAGER       = False
_DEFAULT_ASYNC       = False
_DEFAULT_MAX_IMAGES  = 1
_DEFAULT_MAX_VIDEOS  = 0
_DEFAULT_VLLM_IMAGE  = "nvcr.io/nvidia/vllm:26.07-py3"

_COSMOS3_NANO_MODEL = "nvidia/Cosmos3-Nano"
_COSMOS3_REASONER_ARCHITECTURE = "Cosmos3ForConditionalGeneration"

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
    kv_cache_memory_bytes = cfg.get("kv_cache_memory_bytes")
    if kv_cache_memory_bytes is not None:
        if "gpu_memory_utilization" in cfg:
            logger.error(
                "'kv_cache_memory_bytes' and 'gpu_memory_utilization' are "
                "mutually exclusive"
            )
            sys.exit(1)
        if isinstance(kv_cache_memory_bytes, bool):
            logger.error("'kv_cache_memory_bytes' must be a positive integer")
            sys.exit(1)
        try:
            kv_cache_memory_bytes = int(kv_cache_memory_bytes)
        except (TypeError, ValueError):
            logger.error("'kv_cache_memory_bytes' must be a positive integer")
            sys.exit(1)
        if kv_cache_memory_bytes <= 0:
            logger.error("'kv_cache_memory_bytes' must be a positive integer")
            sys.exit(1)
        gpu_mem = None
    else:
        gpu_mem = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = parse_config_bool(
        cfg.get("enforce_eager", _DEFAULT_EAGER), "enforce_eager"
    )
    async_sched = parse_config_bool(
        cfg.get("async_scheduling", _DEFAULT_ASYNC), "async_scheduling"
    )
    hf_overrides  = cfg.get("hf_overrides")
    mm_encoder_tp_mode = cfg.get("mm_encoder_tp_mode")
    max_images    = int(cfg.get("max_images_per_prompt", _DEFAULT_MAX_IMAGES))
    max_videos    = int(cfg.get("max_videos_per_prompt", _DEFAULT_MAX_VIDEOS))
    backend       = cfg.get("vllm_backend",         "pip")
    image         = cfg.get("vllm_image",           _DEFAULT_VLLM_IMAGE)

    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../models")
    cuda_devices = setup_hf_env(cfg, model_cache)

    if model == _COSMOS3_NANO_MODEL:
        architectures = (
            hf_overrides.get("architectures")
            if isinstance(hf_overrides, dict)
            else None
        )
        if architectures != [_COSMOS3_REASONER_ARCHITECTURE]:
            logger.error(
                "Cosmos3 Nano requires hf_overrides.architectures=[{}] to "
                "guarantee vLLM's Reasoner-only weight-mapping path",
                _COSMOS3_REASONER_ARCHITECTURE,
            )
            sys.exit(1)

    extra_serve_args = [
        "--served-model-name", served_name,
        "--trust-remote-code",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--limit-mm-per-prompt", json.dumps({"image": max_images, "video": max_videos}),
    ]
    if kv_cache_memory_bytes is not None:
        extra_serve_args.extend(
            ["--kv-cache-memory-bytes", str(kv_cache_memory_bytes)]
        )
    else:
        extra_serve_args.extend(["--gpu-memory-utilization", str(gpu_mem)])
    if enforce_eager:
        extra_serve_args.append("--enforce-eager")
    if async_sched:
        extra_serve_args.append("--async-scheduling")
    if hf_overrides:
        extra_serve_args.extend(["--hf-overrides", json.dumps(hf_overrides)])
    if mm_encoder_tp_mode:
        extra_serve_args.extend(["--mm-encoder-tp-mode", str(mm_encoder_tp_mode)])

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
