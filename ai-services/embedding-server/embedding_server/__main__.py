# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
embedding_server — vLLM launcher for nvidia/llama-nemotron-embed-1b-v2.

Reads config and dispatches through ``xr_ai_vllm.serve`` to either the
pip-installed ``vllm`` CLI or the NGC ``nvcr.io/nvidia/vllm`` docker container
(per ``vllm_backend`` in YAML). Runs vLLM with ``--runner pooling --convert
embed`` so it exposes ``/v1/embeddings``; chat/completions is not available.

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
    embedding_dim:           int    Matryoshka truncation dimension for consumers
                                    (metadata only — not passed to vLLM).
    vllm_backend:            str    "pip" (default) or "docker".
    vllm_image:              str    NGC image when vllm_backend=docker
                                    (default: nvcr.io/nvidia/vllm:26.04-py3).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from xr_ai_logging import setup_logging
from xr_ai_vllm import DEFAULT_IMAGE, serve

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


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../../../models")
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("embedding-server")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    model        = cfg.get("model",                _DEFAULT_MODEL)
    host         = cfg.get("host",                 _DEFAULT_HOST)
    port         = int(cfg.get("port",             _DEFAULT_PORT))
    served_name  = cfg.get("served_model_name",    _DEFAULT_SERVED_NAME)
    max_seqs     = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size      = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx      = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem      = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = bool(cfg.get("enforce_eager",  _DEFAULT_EAGER))
    backend      = cfg.get("vllm_backend",         "pip")
    image        = cfg.get("vllm_image",           DEFAULT_IMAGE)

    cuda_devices = cfg.get("cuda_visible_devices")
    if cuda_devices is not None:
        cuda_devices = str(cuda_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HOME"] = str(model_cache)

    extra_serve_args = [
        # vLLM ≥0.10 replaced --task with --runner / --convert.
        # --runner pooling + --convert embed reproduces the old `--task embed`
        # semantics: pooling runner + embedding head, exposes /v1/embeddings.
        "--runner", "pooling",
        "--convert", "embed",
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
        hf_token=hf_token or None,
        cuda_visible_devices=cuda_devices,
        ready_file=ns.ready_file,
    )


if __name__ == "__main__":
    run()
