# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nemotron_omni_llm_server — vLLM launcher for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.

Omni multimodal model (text + video). Selects the weight quantisation based
on GPU compute capability and dispatches through ``xr_ai_vllm.serve`` to either
the pip-installed ``vllm`` CLI or the NGC ``nvcr.io/nvidia/vllm`` docker
container (per ``vllm_backend`` in YAML).

Config keys (nemotron_omni_llm_server.yaml)
-------------------------------------------
    model_blackwell:          str    HF model ID for Blackwell (SM100+) — NVFP4.
    model_ada:                str    HF model ID for Ada / Hopper / Ampere — FP8.
    model_bf16:               str    HF model ID for BF16 (fallback / no quant).
    use_bf16:                 bool   Force BF16 regardless of GPU (default: false).
    host:                     str    Bind address (default: "0.0.0.0").
    port:                     int    HTTP port (default: 8108).
    served_model_name:        str    Name in /v1/models (default: "llm").
    hf_token:                 str    HF token for gated models.
    model_cache:              str    Weight cache, relative to this YAML.
    max_num_seqs:             int    vLLM --max-num-seqs (default: 384).
    tensor_parallel_size:     int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:            int    vLLM --max-model-len (default: 131072).
    gpu_memory_utilization:   float  vLLM --gpu-memory-utilization (default: 0.85).
    enforce_eager:            bool   Skip CUDA graph capture (default: false).
    video_pruning_rate:       float  --video-pruning-rate (default: 0.5).
    video_fps:                int    FPS for video input sampling (default: 2).
    video_num_frames:         int    Max frames per video (default: 256).
    moe_backend:              str    Optional vLLM MoE backend (for example,
                                     "triton" on RTX Pro Blackwell).
    vllm_backend:             str    "pip" (default) or "docker".
    vllm_image:               str    Image when vllm_backend=docker (default:
                                     vllm/vllm-openai:v0.20.0, the minimum
                                     version supporting this architecture).
    extra_pip:                list   Optional packages installed before
                                     `vllm serve` (docker backend only).
"""
import json
import os

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_vllm import (
    gpu_compute_major,
    load_config,
    resolve_model_cache,
    serve,
    setup_hf_env,
)

_MODEL_BLACKWELL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
_MODEL_ADA       = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
_MODEL_BF16      = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

_DEFAULT_PORT    = 8108
_DEFAULT_HOST    = "0.0.0.0"
_DEFAULT_SERVED  = "llm"
_DEFAULT_SEQS    = 384
_DEFAULT_TP      = 1
_DEFAULT_CTX     = 131072
_DEFAULT_GPU_MEM = 0.85
_DEFAULT_EAGER   = False
_DEFAULT_PRUNE   = 0.5
_DEFAULT_FPS     = 2
_DEFAULT_FRAMES  = 256
_DEFAULT_IMAGE   = "vllm/vllm-openai:v0.20.0"

_CONTAINER_NAME = "xr-ai-vllm-nemotron-omni-llm-server"


def run() -> None:
    setup_logging("llm-nemotron-omni")

    cfg, yaml_dir, ready_file = load_config()

    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../models")
    # setup_hf_env sets CUDA_VISIBLE_DEVICES before gpu_compute_major() so
    # nvidia-smi queries the right device.
    cuda_devices = setup_hf_env(cfg, model_cache)

    if cfg.get("use_bf16", False):
        model = cfg.get("model_bf16", _MODEL_BF16)
        use_kv_fp8 = False
        logger.info("use_bf16=true → {}", model)
    else:
        major = gpu_compute_major()
        if major >= 10:
            model = cfg.get("model_blackwell", _MODEL_BLACKWELL)
            use_kv_fp8 = True
            logger.info("Blackwell (SM{}0) → {}", major, model)
        else:
            model = cfg.get("model_ada", _MODEL_ADA)
            use_kv_fp8 = True
            arch = f"SM{major}0" if major > 0 else "unknown GPU"
            logger.info("Pre-Blackwell ({}) → {}", arch, model)

    host          = cfg.get("host",                 _DEFAULT_HOST)
    port          = int(cfg.get("port",             _DEFAULT_PORT))
    served_name   = cfg.get("served_model_name",    _DEFAULT_SERVED)
    max_seqs      = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size       = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx       = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem       = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = bool(cfg.get("enforce_eager",   _DEFAULT_EAGER))
    prune_rate    = float(cfg.get("video_pruning_rate", _DEFAULT_PRUNE))
    video_fps     = int(cfg.get("video_fps",        _DEFAULT_FPS))
    video_frames  = int(cfg.get("video_num_frames", _DEFAULT_FRAMES))
    moe_backend   = cfg.get("moe_backend")
    backend       = cfg.get("vllm_backend",         "pip")
    image         = cfg.get("vllm_image",           _DEFAULT_IMAGE)
    extra_pip     = cfg.get("extra_pip", [])

    media_io_kwargs = json.dumps({"video": {"fps": video_fps, "num_frames": video_frames}})

    extra_serve_args = [
        "--served-model-name", served_name,
        "--trust-remote-code",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", str(gpu_mem),
        "--video-pruning-rate", str(prune_rate),
        "--allowed-local-media-path", "/",
        "--media-io-kwargs", media_io_kwargs,
        "--reasoning-parser", "nemotron_v3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    if use_kv_fp8:
        extra_serve_args += ["--kv-cache-dtype", "fp8"]
    if moe_backend:
        extra_serve_args += ["--moe-backend", str(moe_backend)]
    if enforce_eager:
        extra_serve_args.append("--enforce-eager")

    serve(
        backend=backend,
        persistent=False,
        image=image,
        container_name=_CONTAINER_NAME,
        log_prefix="nemotron_omni",
        model=model,
        extra_serve_args=extra_serve_args,
        host=host,
        port=port,
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN") or None,
        cuda_visible_devices=cuda_devices,
        extra_pip=extra_pip,
        ready_file=ready_file,
    )


if __name__ == "__main__":
    run()
