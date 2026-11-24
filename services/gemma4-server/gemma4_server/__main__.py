# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Gemma 4 conversation, reasoning, and vision with the shared Docker lifecycle."""

from __future__ import annotations

import json
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
        "--served-model-name", str(cfg.get("served_model_name", "gemma4")),
        "--tensor-parallel-size", str(cfg.get("tensor_parallel_size", 1)),
        "--max-model-len", str(cfg.get("max_model_len", 32768)),
        "--max-num-seqs", str(cfg.get("max_num_seqs", 8)),
        "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.5)),
        "--reasoning-parser", "gemma4",
        "--limit-mm-per-prompt", json.dumps({"image": cfg.get("max_images_per_prompt", 1), "audio": 0, "video": 0}),
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
    setup_logging("gemma4-server")
    cfg, yaml_dir, ready_file = load_config()
    model_cache = resolve_model_cache(cfg, yaml_dir, default="../../models")
    cuda_devices = setup_hf_env(cfg, model_cache)
    spark_uma = cfg["spark_uma"] if "spark_uma" in cfg else gpu_is_dgx_spark()
    serve(
        backend="docker",
        persistent=True,
        image=cfg.get("vllm_image", "vllm/vllm-openai:v0.33.0"),
        container_name="xr-ai-vllm-gemma4-server",
        log_prefix="gemma4_server",
        model=cfg.get("model", "google/gemma-4-E2B-it"),
        extra_serve_args=_serve_args(cfg),
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8108)),
        model_cache=model_cache,
        hf_token=os.environ.get("HF_TOKEN") or None,
        cuda_visible_devices=cuda_devices,
        ready_file=ready_file,
        spark_uma=parse_config_bool(spark_uma, "spark_uma"),
    )


if __name__ == "__main__":
    run()
