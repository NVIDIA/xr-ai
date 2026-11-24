# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve D3 typed decisions with the shared Docker lifecycle."""

from __future__ import annotations

import os

from xr_ai_logging import setup_logging
from xr_ai_vllm import (
    gpu_is_dgx_spark,
    load_config,
    parse_config_bool,
    resolve_model_cache,
    serve,
    setup_hf_env,
)


def _serve_args(cfg: dict) -> list[str]:
    args = [
        "--served-model-name", str(cfg.get("served_model_name", "d3")),
        "--tensor-parallel-size", str(cfg.get("tensor_parallel_size", 1)),
        "--max-model-len", str(cfg.get("max_model_len", 8192)),
        "--max-num-seqs", str(cfg.get("max_num_seqs", 8)),
        "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.5)),
        # D3 reads full attention bidirectionally. Its native runner supplies
        # SystemOne classification and multimodal preprocessing automatically.
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
    ]
    if "kv_cache_memory_bytes" in cfg:
        value = cfg["kv_cache_memory_bytes"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("kv_cache_memory_bytes must be a positive integer")
        args.extend(["--kv-cache-memory-bytes", str(value)])
    if parse_config_bool(cfg.get("enforce_eager", False), "enforce_eager"):
        args.append("--enforce-eager")
    return args


def run() -> None:
    setup_logging("d3-server")
    cfg, yaml_dir, ready_file = load_config()
    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../models")
    cuda_devices = setup_hf_env(cfg, model_cache)
    spark_uma = cfg["spark_uma"] if "spark_uma" in cfg else gpu_is_dgx_spark()
    serve(
        backend="docker",
        persistent=True,
        image=cfg.get("vllm_image", "vllm/vllm-openai:v0.33.0"),
        container_name="xr-ai-vllm-d3-server",
        log_prefix="d3_server",
        model=cfg.get("model", "vllm-sr/d3-nano"),
        extra_serve_args=_serve_args(cfg),
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8120)),
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN") or None,
        cuda_visible_devices=cuda_devices,
        ready_file=ready_file,
        spark_uma=parse_config_bool(spark_uma, "spark_uma"),
    )


if __name__ == "__main__":
    run()
