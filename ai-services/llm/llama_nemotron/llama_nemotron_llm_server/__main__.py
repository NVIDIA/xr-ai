# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
llama_nemotron_llm_server — vLLM launcher for Llama-3.1-Nemotron-Nano-8B-v1.

Reads config and dispatches through ``xr_ai_vllm.serve`` to either the
pip-installed ``vllm`` CLI or the NGC ``nvcr.io/nvidia/vllm`` docker container
(per ``vllm_backend`` in YAML). vLLM handles the OpenAI-compatible HTTP API,
the native Llama-3.1 chat template, and tool-call parsing.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:                   str    HuggingFace model ID.
    host:                    str    Bind address (default: "0.0.0.0").
    port:                    int    HTTP port (default: 8106).
    served_model_name:       str    Name exposed in /v1/models (default: "llm").
    hf_token:                str    HuggingFace token for gated models.
    model_cache:             str    HF weight cache, relative to this YAML.
    max_num_seqs:            int    vLLM --max-num-seqs (default: 8).
    tensor_parallel_size:    int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:           int    vLLM --max-model-len (default: 32768).
    gpu_memory_utilization:  float  vLLM --gpu-memory-utilization (default: 0.85).
    enforce_eager:           bool   Skip CUDA graph capture (default: false).
    tool_call_parser:        str    vLLM --tool-call-parser (default: "llama3_json").
    enable_tool_choice:      bool   Pass --enable-auto-tool-choice (default: true).
    vllm_backend:            str    "pip" (default) or "docker".
    vllm_image:              str    NGC image when vllm_backend=docker
                                    (default: nvcr.io/nvidia/vllm:26.04-py3).
"""
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

_DEFAULT_PORT               = 8106
_DEFAULT_HOST               = "0.0.0.0"
_DEFAULT_SERVED_NAME        = "llm"
_DEFAULT_SEQS               = 8
_DEFAULT_TP                 = 1
_DEFAULT_CTX                = 32768
_DEFAULT_GPU_MEM            = 0.85
_DEFAULT_EAGER              = False
_DEFAULT_TOOL_CALL_PARSER   = "llama3_json"
_DEFAULT_ENABLE_TOOL_CHOICE = True

_CONTAINER_NAME = "xr-ai-vllm-llama-nemotron-llm-server"


def run() -> None:
    setup_logging("llm-llama-nemotron")

    cfg, yaml_dir, ready_file = load_config()

    if not cfg.get("model"):
        logger.error("'model' is required in config")
        sys.exit(1)

    model              = cfg["model"]
    host               = cfg.get("host",                 _DEFAULT_HOST)
    port               = int(cfg.get("port",             _DEFAULT_PORT))
    served_name        = cfg.get("served_model_name",    _DEFAULT_SERVED_NAME)
    max_seqs           = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size            = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx            = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem            = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager      = bool(cfg.get("enforce_eager",   _DEFAULT_EAGER))
    tool_call_parser   = cfg.get("tool_call_parser",     _DEFAULT_TOOL_CALL_PARSER)
    enable_tool_choice = bool(cfg.get("enable_tool_choice", _DEFAULT_ENABLE_TOOL_CHOICE))
    backend            = cfg.get("vllm_backend",         "pip")
    image              = cfg.get("vllm_image",           DEFAULT_IMAGE)

    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../../models")
    cuda_devices = setup_hf_env(cfg, model_cache)

    extra_serve_args = [
        "--served-model-name", served_name,
        "--trust-remote-code",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if enable_tool_choice:
        extra_serve_args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
    if enforce_eager:
        extra_serve_args.append("--enforce-eager")

    serve(
        backend=backend,
        persistent=True,
        image=image,
        container_name=_CONTAINER_NAME,
        log_prefix="llama_nemotron",
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
