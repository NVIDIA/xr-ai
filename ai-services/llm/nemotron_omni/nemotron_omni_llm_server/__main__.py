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
import logging
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


def _resolve_log_level(cfg: dict) -> str:
    """Per-process YAML log_level > XR_AI_LOG_LEVEL env > INFO. Inlined to
    keep workers stdlib-only and to avoid importing from xr_ai_launcher
    (forbidden for workers per AGENTS.md)."""
    val = cfg.get("log_level")
    if val and isinstance(val, str):
        v = val.upper()
        if v in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
            return v
    env = os.environ.get("XR_AI_LOG_LEVEL", "").upper()
    if env in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
        return env
    return "INFO"


def _resolve_vllm_log_level(cfg: dict) -> str:
    """Per-process YAML vllm_log_level > INFO. vLLM honors VLLM_LOGGING_LEVEL
    env var at startup; we set it BEFORE os.execvp so vLLM's logger inherits
    it across the process replacement (and propagates to vLLM's child
    APIServer + EngineCore workers). Independent of the main `log_level`
    field — that one is gone after execvp."""
    val = cfg.get("vllm_log_level")
    if val and isinstance(val, str):
        v = val.upper()
        if v in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
            return v
    return "INFO"


def _resolve_http_log_level(cfg: dict) -> str:
    """Per-process YAML http_log_level > WARNING. Controls httpx + httpcore
    loggers (per-HTTP-request noise). Independent of the main `log_level`
    field; file capture is unaffected (DEBUG always)."""
    val = cfg.get("http_log_level")
    if val and isinstance(val, str):
        v = val.upper()
        if v in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
            return v
    return "WARNING"


# ── per-source level filters (file gets DEBUG always; terminal at user level) ─

class _AboveThresholdFilter(logging.Filter):
    """Pass records at level >= per-source threshold (terminal display)."""
    def __init__(self, default: int, sources: dict | None = None) -> None:
        super().__init__()
        self.default = default
        self.sources = sources or {}
    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, thr in self.sources.items():
            if record.name.startswith(prefix):
                return record.levelno >= thr
        return record.levelno >= self.default


class _BelowThresholdFilter(logging.Filter):
    """Pass records at level < per-source threshold (file capture for the
    levels the StreamHandler dropped)."""
    def __init__(self, default: int, sources: dict | None = None) -> None:
        super().__init__()
        self.default = default
        self.sources = sources or {}
    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, thr in self.sources.items():
            if record.name.startswith(prefix):
                return record.levelno < thr
        return record.levelno < self.default


def _setup_logging(cfg: dict, sources: dict | None = None) -> None:
    """Multi-handler setup: terminal at user level (per-source-aware via
    AboveThresholdFilter), file at DEBUG via FileHandlers with the inverse
    BelowThresholdFilter — exclusive routing, no duplicates with the
    launcher's PIPE tee.  Reads XR_AI_LOG_DIR + XR_AI_LOG_NAME env vars
    (set by the launcher) to pick file paths; degrades to terminal-only
    when unset."""
    user_level = getattr(logging, _resolve_log_level(cfg), logging.INFO)
    sources = sources or {}
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    sh.addFilter(_AboveThresholdFilter(user_level, sources))
    root.addHandler(sh)

    log_dir  = os.environ.get("XR_AI_LOG_DIR")
    log_name = os.environ.get("XR_AI_LOG_NAME")
    if log_dir and log_name:
        for path in (f"{log_dir}/{log_name}.log", f"{log_dir}/combined.log"):
            try:
                fh = logging.FileHandler(path, mode="a")
            except OSError:
                continue
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            fh.addFilter(_BelowThresholdFilter(user_level, sources))
            root.addHandler(fh)


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

    http_level = getattr(logging, _resolve_http_log_level(cfg), logging.WARNING)
    _setup_logging(cfg, sources={"httpx": http_level, "httpcore": http_level})

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
    os.environ["VLLM_LOGGING_LEVEL"] = _resolve_vllm_log_level(cfg)

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
    logging.shutdown()  # flush FileHandler buffers before the Python image is replaced
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    run()
