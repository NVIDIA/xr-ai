# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nemotron_omni_llm_server — vLLM launcher for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.

Omni multimodal model (text + video). Selects the weight quantisation based
on GPU compute capability and execs into ``vllm serve``.

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
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

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


def _gpu_compute_major() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if out:
            return int(out[0].split(".")[0])
    except Exception:
        # Detection is best-effort; if nvidia-smi is unavailable or parsing fails,
        # fall back to 0 (unknown capability) so caller can select a safe default.
        pass
    return 0


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../../models")
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

    # Model selection
    if cfg.get("use_bf16", False):
        model = cfg.get("model_bf16", _MODEL_BF16)
        use_kv_fp8 = False
        print(f"[nemotron_omni] use_bf16=true → {model}", flush=True)
    else:
        major = _gpu_compute_major()
        if major >= 10:
            model = cfg.get("model_blackwell", _MODEL_BLACKWELL)
            use_kv_fp8 = True
            print(f"[nemotron_omni] Blackwell (SM{major}0) → {model}", flush=True)
        else:
            model = cfg.get("model_ada", _MODEL_ADA)
            use_kv_fp8 = True
            arch = f"SM{major}0" if major > 0 else "unknown GPU"
            print(f"[nemotron_omni] Pre-Blackwell ({arch}) → {model}", flush=True)

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

    if cuda_devices := cfg.get("cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    if hf_token := (cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")):
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HOME"] = str(model_cache)

    media_io_kwargs = json.dumps({"video": {"fps": video_fps, "num_frames": video_frames}})

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
        "--video-pruning-rate", str(prune_rate),
        "--allowed-local-media-path", "/",
        "--media-io-kwargs", media_io_kwargs,
        "--reasoning-parser", "nemotron_v3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    if use_kv_fp8:
        argv.extend(["--kv-cache-dtype", "fp8"])
    if enforce_eager:
        argv.append("--enforce-eager")

    print(f"[nemotron_omni] Launching vLLM  http://{host}:{port}/v1  model={model}", flush=True)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    run()
