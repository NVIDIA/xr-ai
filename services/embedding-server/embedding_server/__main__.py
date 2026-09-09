# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
embedding_server — vLLM launcher for nvidia/llama-nemotron-embed-1b-v2.

Reads config and dispatches through ``xr_ai_vllm.serve`` to either the
pip-installed ``vllm`` CLI or the NGC ``nvcr.io/nvidia/vllm`` docker container
(per ``vllm_backend`` in YAML). The model config selects vLLM's pooling runner
and exposes ``/v1/embeddings``; chat/completions is not available.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:                   str    HuggingFace model ID.
    host:                    str    Bind address (default: "0.0.0.0").
    port:                    int    HTTP port (default: 8109).
    served_model_name:       str    Name exposed in /v1/models (default: "embed").
    hf_token:                str    HuggingFace token for gated models.
    model_cache:             str    HF weight cache, relative to this YAML.
    max_num_seqs:            int    vLLM --max-num-seqs (default: 32).
    tensor_parallel_size:    int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:           int    vLLM --max-model-len (default: 8192).
    gpu_memory_utilization:  float  vLLM --gpu-memory-utilization (default: 0.20).
    enforce_eager:           bool   Skip CUDA graph capture (default: false).
    vllm_backend:            str    "pip" (default) or "docker".
    vllm_image:              str    NGC image when vllm_backend=docker
                                    (default: nvcr.io/nvidia/vllm:26.04-py3).
    spark_uma:               bool   Enable DGX Spark cold-start safeguards
                                    (docker backend only; default: false).
"""
import os
import sys

from xr_ai_logging import setup_logging
from xr_ai_vllm import (
    DEFAULT_IMAGE,
    load_config,
    resolve_model_cache,
    serve,
    setup_hf_env,
)
from xr_ai_vllm._config import parse_config_bool

_DEFAULT_MODEL       = "nvidia/llama-nemotron-embed-1b-v2"
_DEFAULT_PORT        = 8109
_DEFAULT_HOST        = "0.0.0.0"
_DEFAULT_SERVED_NAME = "embed"
_DEFAULT_SEQS        = 32
_DEFAULT_TP          = 1
_DEFAULT_CTX         = 8192
_DEFAULT_GPU_MEM     = 0.20
_DEFAULT_EAGER       = False

_CONTAINER_NAME = "xr-ai-vllm-embedding-server"


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("embedding-server")

    cfg, yaml_dir, ready_file = load_config()

    model        = cfg.get("model",                _DEFAULT_MODEL)
    host         = cfg.get("host",                 _DEFAULT_HOST)
    port         = int(cfg.get("port",             _DEFAULT_PORT))
    served_name  = cfg.get("served_model_name",    _DEFAULT_SERVED_NAME)
    max_seqs     = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size      = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx      = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem      = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = parse_config_bool(
        cfg.get("enforce_eager", _DEFAULT_EAGER), "enforce_eager"
    )
    backend      = cfg.get("vllm_backend",         "pip")
    image        = cfg.get("vllm_image",           DEFAULT_IMAGE)
    spark_uma    = parse_config_bool(cfg.get("spark_uma", False), "spark_uma")

    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../models")
    cuda_devices = setup_hf_env(cfg, model_cache)

    extra_serve_args = [
        "--served-model-name", served_name,
        "--trust-remote-code",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if enforce_eager:
        extra_serve_args.append("--enforce-eager")

    serve(
        backend=backend,
        persistent=True,
        image=image,
        container_name=_CONTAINER_NAME,
        log_prefix="embedding_server",
        model=model,
        extra_serve_args=extra_serve_args,
        host=host,
        port=port,
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN") or None,
        cuda_visible_devices=cuda_devices,
        ready_file=ready_file,
        spark_uma=spark_uma,
    )


if __name__ == "__main__":
    run()
