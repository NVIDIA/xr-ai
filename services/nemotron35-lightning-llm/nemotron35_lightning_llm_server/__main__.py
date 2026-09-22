# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Nemotron 3.5 Lightning through an OpenAI-compatible vLLM server."""

from __future__ import annotations

import os

from xr_ai_logging import setup_logging
from xr_ai_vllm import (
    DEFAULT_IMAGE,
    load_config,
    resolve_model_cache,
    serve,
    setup_hf_env,
)
from xr_ai_vllm._config import parse_config_bool

_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
_CONTAINER_NAME = "xr-ai-vllm-nemotron35-lightning-llm-server"


def _optional_arg(args: list[str], flag: str, value: object) -> None:
    if value is not None:
        args.extend((flag, str(value)))


def run() -> None:
    """Load YAML configuration and serve Nemotron 3.5 Lightning."""
    setup_logging("llm-nemotron35-lightning")
    cfg, yaml_dir, ready_file = load_config()

    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../models")
    cuda_devices = setup_hf_env(cfg, model_cache)

    host = str(cfg.get("host", "0.0.0.0"))
    port = int(cfg.get("port", 8108))
    model = str(cfg.get("model", _MODEL))
    served_name = str(cfg.get("served_model_name", "llm"))
    backend = str(cfg.get("vllm_backend", "pip"))
    image = str(cfg.get("vllm_image", DEFAULT_IMAGE))

    args = [
        "--served-model-name",
        served_name,
        "--max-num-seqs",
        str(cfg.get("max_num_seqs", 8)),
        "--tensor-parallel-size",
        str(cfg.get("tensor_parallel_size", 1)),
        "--max-model-len",
        str(cfg.get("max_model_len", 32768)),
        "--gpu-memory-utilization",
        str(cfg.get("gpu_memory_utilization", 0.85)),
        "--reasoning-parser",
        "nemotron_v3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    if parse_config_bool(cfg.get("enable_prefix_caching", True), "enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if parse_config_bool(cfg.get("async_scheduling", False), "async_scheduling"):
        args.append("--async-scheduling")
    if parse_config_bool(cfg.get("enforce_eager", False), "enforce_eager"):
        args.append("--enforce-eager")

    _optional_arg(args, "--kv-cache-dtype", cfg.get("kv_cache_dtype", "fp8"))
    _optional_arg(args, "--kv-cache-memory-bytes", cfg.get("kv_cache_memory_bytes"))
    _optional_arg(args, "--quantization", cfg.get("quantization"))
    _optional_arg(args, "--mamba-backend", cfg.get("mamba_backend", "flashinfer"))
    _optional_arg(args, "--mamba-cache-mode", cfg.get("mamba_cache_mode", "align"))
    _optional_arg(args, "--mamba-ssu-algorithm", cfg.get("mamba_ssu_algorithm"))
    moe_backend = cfg.get("moe_backend")
    _optional_arg(args, "--moe-backend", moe_backend)
    _optional_arg(args, "--linear-backend", cfg.get("linear_backend"))

    extra_env = (
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "indexed"}
        if moe_backend == "humming"
        else {}
    )
    if backend == "pip":
        os.environ.update(extra_env)

    serve(
        backend=backend,
        persistent=True,
        image=image,
        container_name=_CONTAINER_NAME,
        log_prefix="nemotron35_lightning",
        model=model,
        extra_serve_args=args,
        host=host,
        port=port,
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN") or None,
        cuda_visible_devices=cuda_devices,
        extra_env=extra_env,
        ready_file=ready_file,
        spark_uma=parse_config_bool(cfg.get("spark_uma", False), "spark_uma"),
    )


if __name__ == "__main__":
    run()
