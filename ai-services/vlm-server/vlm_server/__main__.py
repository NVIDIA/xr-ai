# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vlm_server — vLLM launcher for Cosmos-Reason1-7B (or any Qwen2.5-VL-compatible VLM).

Reads config, sets HuggingFace env vars, and execs into ``vllm serve``.
vLLM handles the OpenAI-compatible HTTP API, image decoding, and weight loading.

Images are passed as base64 data URLs in the ``image_url`` content block,
same as the OpenAI vision API format.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:                   str    HuggingFace model ID.
    host:                    str    Bind address (default: "0.0.0.0").
    port:                    int    HTTP port (default: 8100).
    served_model_name:       str    Name exposed in /v1/models (default: "vlm").
    hf_token:                str    HuggingFace token for gated models.
    model_cache:             str    HF weight cache, relative to this YAML.
    max_num_seqs:            int    vLLM --max-num-seqs (default: 4).
    tensor_parallel_size:    int    vLLM --tensor-parallel-size (default: 1).
    max_model_len:           int    vLLM --max-model-len (default: 8192).
    gpu_memory_utilization:  float  vLLM --gpu-memory-utilization (default: 0.85).
    enforce_eager:           bool   Skip CUDA graph capture (default: false).
    max_images_per_prompt:   int    Max images per request (default: 1).
    max_videos_per_prompt:   int    Max video items per request (default: 0).
                                    Set >0 only if your worker sends video;
                                    0 skips vLLM's video activation profiling
                                    at startup, saving tens of GiB on
                                    Qwen2.5-VL-class models.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

_DEFAULT_MODEL       = "nvidia/Cosmos-Reason1-7B"
_DEFAULT_PORT        = 8100
_DEFAULT_HOST        = "0.0.0.0"
_DEFAULT_SERVED_NAME = "vlm"
_DEFAULT_SEQS        = 4
_DEFAULT_TP          = 1
_DEFAULT_CTX         = 8192
_DEFAULT_GPU_MEM     = 0.85
_DEFAULT_EAGER       = False
_DEFAULT_MAX_IMAGES  = 1
_DEFAULT_MAX_VIDEOS  = 0


def _idle_until_stopped(health_url: str, poll_s: float = 5.0) -> None:
    """Block until the health endpoint stops responding or a signal arrives."""
    stopped = [False]
    orig_term = signal.getsignal(signal.SIGTERM)
    orig_int  = signal.getsignal(signal.SIGINT)

    def _on_signal(sig, _frame):
        stopped[0] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)
    try:
        while not stopped[0]:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as r:
                    if r.status != 200:
                        print("[vlm_server] existing vLLM stopped responding — exiting",
                              flush=True)
                        return
            except Exception:
                print("[vlm_server] existing vLLM unreachable — exiting", flush=True)
                return
            time.sleep(poll_s)
    finally:
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT,  orig_int)


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../models")
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


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

    model         = cfg.get("model",               _DEFAULT_MODEL)
    host          = cfg.get("host",                 _DEFAULT_HOST)
    port          = int(cfg.get("port",             _DEFAULT_PORT))
    served_name   = cfg.get("served_model_name",    _DEFAULT_SERVED_NAME)
    max_seqs      = int(cfg.get("max_num_seqs",     _DEFAULT_SEQS))
    tp_size       = int(cfg.get("tensor_parallel_size", _DEFAULT_TP))
    max_ctx       = int(cfg.get("max_model_len",    _DEFAULT_CTX))
    gpu_mem       = float(cfg.get("gpu_memory_utilization", _DEFAULT_GPU_MEM))
    enforce_eager = bool(cfg.get("enforce_eager",   _DEFAULT_EAGER))
    max_images    = int(cfg.get("max_images_per_prompt", _DEFAULT_MAX_IMAGES))
    max_videos    = int(cfg.get("max_videos_per_prompt", _DEFAULT_MAX_VIDEOS))

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
        "--limit-mm-per-prompt", json.dumps({"image": max_images, "video": max_videos}),
    ]
    if enforce_eager:
        argv.append("--enforce-eager")

    health_url = f"http://127.0.0.1:{port}/health"

    # Reuse an already-running vLLM instance (e.g. survived a worker crash).
    try:
        with urllib.request.urlopen(health_url, timeout=3) as r:
            already_up = r.status == 200
    except Exception:
        already_up = False

    if already_up:
        print(f"[vlm_server] vLLM already running on port {port} — reusing", flush=True)
        if ns.ready_file:
            ns.ready_file.touch()
        _idle_until_stopped(health_url)
        return

    print(f"[vlm_server] Launching vLLM  http://{host}:{port}/v1  model={model}", flush=True)
    # start_new_session=True puts vLLM in its own process group so the
    # launcher's killpg() does not reach it when shutting down the wrapper.
    proc = subprocess.Popen(argv, start_new_session=True)

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

    print(f"[vlm_server] Ready  →  http://localhost:{port}/v1", flush=True)
    if ns.ready_file:
        ns.ready_file.touch()

    proc.wait()


if __name__ == "__main__":
    run()
