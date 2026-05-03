# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
mistral_minitron_llm_server — vLLM launcher for Mistral-NeMo-Minitron-8B-Instruct.

Reads config, sets HuggingFace env vars, and execs into ``vllm serve``.
vLLM handles the OpenAI-compatible HTTP API and weight loading.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:                   str    HuggingFace model ID.
    host:                    str    Bind address (default: "0.0.0.0").
    port:                    int    HTTP port (default: 8101).
    served_model_name:       str    Name exposed in /v1/models (default: "llm").
    hf_token:                str    HuggingFace token for gated models.
    model_cache:             str    HF weight cache, relative to this YAML.
    max_num_seqs:            int    vLLM --max-num-seqs (default: 8).
    tensor_parallel_size:    int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:           int    vLLM --max-model-len (default: 32768).
    gpu_memory_utilization:  float  vLLM --gpu-memory-utilization (default: 0.85).
    enforce_eager:           bool   Skip CUDA graph capture (default: false).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

_DEFAULT_MODEL       = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
_DEFAULT_PORT        = 8101
_DEFAULT_HOST        = "0.0.0.0"
_DEFAULT_SERVED_NAME = "llm"
_DEFAULT_SEQS        = 8
_DEFAULT_TP          = 1
_DEFAULT_CTX         = 32768
_DEFAULT_GPU_MEM     = 0.85
_DEFAULT_EAGER       = False


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

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    model         = cfg.get("model",               _DEFAULT_MODEL)
    host          = cfg.get("host",                 _DEFAULT_HOST)
    port          = int(cfg.get("port",             _DEFAULT_PORT))
    served_name   = cfg.get("served_model_name",    _DEFAULT_SERVED_NAME)
    max_seqs      = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size       = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx       = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem       = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = bool(cfg.get("enforce_eager",   _DEFAULT_EAGER))

    if cuda_devices := cfg.get("cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    if hf_token := (cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")):
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HOME"] = str(model_cache)

    argv = [
        "vllm", "serve", model,
        "--served-model-name", served_name,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if enforce_eager:
        argv.append("--enforce-eager")

    print(f"[mistral_minitron] Launching vLLM  http://{host}:{port}/v1  model={model}", flush=True)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    run()
