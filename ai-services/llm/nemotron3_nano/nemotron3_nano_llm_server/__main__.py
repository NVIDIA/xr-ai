# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nemotron3_nano_llm_server — vLLM launcher for Nemotron-3-Nano-30B.

Selects the FP8 or NVFP4 model variant based on GPU compute capability,
downloads the nano_v3 reasoning-parser plugin once, and execs into
``vllm serve``.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model_blackwell:         str    HF model ID for Blackwell (SM100+).
    model_ada:               str    HF model ID for Ada / Hopper / Ampere.
    host:                    str    Bind address (default: "0.0.0.0").
    port:                    int    HTTP port (default: 8107).
    served_model_name:       str    Name exposed in /v1/models (default: "llm").
    hf_token:                str    HuggingFace token for gated models.
    model_cache:             str    HF weight + plugin cache, relative to this YAML.
    max_num_seqs:            int    vLLM --max-num-seqs (default: 8).
    tensor_parallel_size:    int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:           int    vLLM --max-model-len (default: 32768).
    gpu_memory_utilization:  float  vLLM --gpu-memory-utilization (default: 0.85).
    enforce_eager:           bool   Skip CUDA graph capture (default: true).
    parser_url:              str    URL to fetch nano_v3_reasoning_parser.py.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

_MODEL_BLACKWELL  = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
_MODEL_ADA        = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
_DEFAULT_PORT     = 8107
_DEFAULT_HOST     = "0.0.0.0"
_DEFAULT_SERVED   = "llm"
_DEFAULT_SEQS     = 8
_DEFAULT_TP       = 1
_DEFAULT_CTX      = 32768
_DEFAULT_GPU_MEM  = 0.85
_DEFAULT_EAGER    = True

_PARSER_FILENAME = "nano_v3_reasoning_parser.py"
_PARSER_URL_DEFAULT = (
    "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    f"/resolve/main/{_PARSER_FILENAME}"
)


def _gpu_compute_major() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if out:
            return int(out[0].split(".")[0])
    except Exception:
        pass
    return 0


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../../models")
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_reasoning_parser(model_cache: Path, url: str) -> Path:
    path = model_cache / _PARSER_FILENAME
    if not path.exists():
        print(f"[nemotron3_nano] Downloading {_PARSER_FILENAME}…", flush=True)
        with urllib.request.urlopen(url) as resp:
            path.write_bytes(resp.read())
    return path


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

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

    # Set before _gpu_compute_major() so nvidia-smi queries the right device.
    if cuda_devices := cfg.get("cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    major = _gpu_compute_major()
    if major >= 10:
        model = cfg.get("model_blackwell", _MODEL_BLACKWELL)
        print(f"[nemotron3_nano] Blackwell (SM{major}0) → {model}", flush=True)
    else:
        model = cfg.get("model_ada", _MODEL_ADA)
        arch  = f"SM{major}0" if major > 0 else "unknown GPU"
        print(f"[nemotron3_nano] Pre-Blackwell ({arch}) → {model}", flush=True)

    host          = cfg.get("host",               _DEFAULT_HOST)
    port          = int(cfg.get("port",            _DEFAULT_PORT))
    served_name   = cfg.get("served_model_name",   _DEFAULT_SERVED)
    max_seqs      = int(cfg.get("max_num_seqs",    _DEFAULT_SEQS))
    tp_size       = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx       = int(cfg.get("max_model_len",   _DEFAULT_CTX))
    gpu_mem       = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = bool(cfg.get("enforce_eager",  _DEFAULT_EAGER))
    parser_url    = cfg.get("parser_url",          _PARSER_URL_DEFAULT)

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    if hf_token := (cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")):
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HOME"] = str(model_cache)

    # FlashInfer JIT-compiles CUTLASS MoE kernels on first run via nvcc.
    # Ensure nvcc is on PATH (CUDA toolkit may not be in the login shell's PATH
    # even when CUDA is installed) and cap ninja parallelism so concurrent nvcc
    # processes don't exhaust RAM on unified-memory machines like DGX Spark.
    _cuda_bin = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
    if _cuda_bin.exists():
        os.environ["PATH"] = str(_cuda_bin) + ":" + os.environ.get("PATH", "")
    os.environ.setdefault("MAX_JOBS", "4")

    parser_path = _ensure_reasoning_parser(model_cache, parser_url)

    argv = [
        "vllm", "serve", model,
        "--served-model-name", served_name,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser-plugin", str(parser_path),
        "--reasoning-parser", "nano_v3",
        "--max-num-seqs", str(max_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if major >= 10:
        argv.extend(["--kv-cache-dtype", "fp8"])
    if enforce_eager:
        argv.append("--enforce-eager")

    print(f"[nemotron3_nano] Launching vLLM  http://{host}:{port}/v1  model={model}", flush=True)
    proc = subprocess.Popen(argv)

    def _fwd(sig, _frame):
        proc.send_signal(sig)

    signal.signal(signal.SIGTERM, _fwd)
    signal.signal(signal.SIGINT,  _fwd)

    health_url = f"http://127.0.0.1:{port}/health"
    while True:
        if proc.poll() is not None:
            sys.exit(proc.returncode or 1)
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(2)

    print(f"[nemotron3_nano] Ready  →  http://localhost:{port}/v1", flush=True)
    if ns.ready_file:
        ns.ready_file.touch()

    proc.wait()


if __name__ == "__main__":
    run()
