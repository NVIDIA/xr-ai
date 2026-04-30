# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nemotron3_nano_llm_server — thin launcher for vLLM serving
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.

vLLM already exposes an OpenAI-compatible HTTP API and handles the three
things our ``llama_nemotron`` server implements by hand for Llama-3.1-Nemotron:

- **Tool calling** — ``--enable-auto-tool-choice --tool-call-parser qwen3_coder``
  parses Nemotron-3-Nano's native ``<tool_call><function=...><parameter=...>``
  XML and emits OpenAI-compatible ``tool_calls`` in the response.
- **Reasoning extraction** — ``--reasoning-parser nano_v3`` (custom plugin from
  the model card) splits the model's ``<think>…</think>`` preamble into
  ``message.reasoning_content`` so ``message.content`` stays clean for TTS.
- **Blackwell FP4 kernels** — ``VLLM_USE_FLASHINFER_MOE_FP4=1`` invokes
  FlashInfer's hardware FP4 MoE kernels on B200 / RTX PRO 6000 / Jetson Thor.

This script is a ~60-line shim: read the YAML config, fetch the reasoning
parser plugin if needed, set the FlashInfer env vars, and ``execvp`` into
``vllm serve`` so the launcher's signals propagate naturally to the
inference server.

Accepts ``--config <path>.yaml`` (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:                   str    HuggingFace model ID (default:
                                    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
    host:                    str    Bind address (default: "0.0.0.0")
    port:                    int    HTTP port (default: 8107)
    hf_token:                str    HuggingFace token for gated models
    model_cache:             str    HF weight + reasoning-parser cache.
                                    Resolved relative to the YAML file.
                                    Default: ../../models
    max_num_seqs:            int    vLLM ``--max-num-seqs``   (default: 8)
    tensor_parallel_size:    int    vLLM ``--tensor-parallel-size`` (default: 1)
    max_model_len:           int    vLLM ``--max-model-len``  (default: 32768)
    gpu_memory_utilization:  float  vLLM ``--gpu-memory-utilization``
                                    (default: 0.6).  Lowered from vLLM's
                                    default of 0.92 because this LLM shares
                                    the GPU with vlm-server (Cosmos-Reason1-7B,
                                    ~14 GB) and stt-server (~1.5 GB) in the
                                    pipecat-nat-nemotron3nano sample.  At
                                    0.6 on a 95 GB Blackwell card vLLM
                                    reserves ~57 GB, leaving ~38 GB for
                                    the other services.
    enforce_eager:           bool   vLLM ``--enforce-eager`` (default: True).
                                    Skips CUDA graph capture + FlashInfer
                                    autotune entirely.  For NemotronH-30B-A3B-
                                    NVFP4 with 128 MoE experts + Mamba-2
                                    layers, graph capture silently takes
                                    3-8 min on first run, producing no log
                                    output and making the worker appear
                                    stuck.  Eager mode starts in ~5 s after
                                    weight load and is 10-20%% slower on
                                    steady-state inference — negligible
                                    for a voice agent (<250 tokens per
                                    turn, STT+VAD+TTS already dominate).
                                    Set to False if you need maximum
                                    throughput and can wait the initial
                                    capture time.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

import yaml

_DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
_DEFAULT_PORT = 8107
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_MAX_NUM_SEQS = 8
_DEFAULT_TP_SIZE = 1
_DEFAULT_MAX_MODEL_LEN = 32768
# Lowered from vLLM's own default of 0.92 because this LLM shares the
# GPU with vlm-server (~14 GB) and stt-server (~1.5 GB).  0.6 on a
# 95 GB Blackwell card gives vLLM ~57 GB and leaves ~38 GB for others.
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.6
# Default enforce_eager=True because NemotronH-30B-A3B-NVFP4's CUDA
# graph capture + FlashInfer MoE autotune is silent and slow (3-8 min
# on first run) — bad UX for a voice agent.  See docstring.
_DEFAULT_ENFORCE_EAGER = True

# Custom reasoning-parser plugin shipped as a raw .py alongside the model
# weights on HuggingFace.  vLLM's ``--reasoning-parser-plugin`` expects a
# filesystem path, so we download once into the shared model_cache and
# reuse it on subsequent runs.
_PARSER_FILENAME = "nano_v3_reasoning_parser.py"
_PARSER_URL = (
    f"https://huggingface.co/{_DEFAULT_MODEL}/resolve/main/{_PARSER_FILENAME}"
)


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../../models")
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_reasoning_parser(model_cache: Path) -> Path:
    """Fetch ``nano_v3_reasoning_parser.py`` once; return its on-disk path."""
    path = model_cache / _PARSER_FILENAME
    if not path.exists():
        print(
            f"[nemotron3_nano_llm_server] Downloading {_PARSER_FILENAME} "
            f"from HuggingFace…",
            flush=True,
        )
        try:
            with urllib.request.urlopen(_PARSER_URL) as resp:
                path.write_bytes(resp.read())
        except Exception as exc:
            raise RuntimeError(
                f"failed to download {_PARSER_URL}: {exc}"
            ) from exc
    return path


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

    model = cfg.get("model", _DEFAULT_MODEL)
    host = cfg.get("host", _DEFAULT_HOST)
    port = int(cfg.get("port", _DEFAULT_PORT))
    max_num_seqs = int(cfg.get("max_num_seqs", _DEFAULT_MAX_NUM_SEQS))
    tp_size = int(cfg.get("tensor_parallel_size", _DEFAULT_TP_SIZE))
    max_model_len = int(cfg.get("max_model_len", _DEFAULT_MAX_MODEL_LEN))
    gpu_mem_util = float(cfg.get(
        "gpu_memory_utilization", _DEFAULT_GPU_MEMORY_UTILIZATION,
    ))
    enforce_eager = bool(cfg.get("enforce_eager", _DEFAULT_ENFORCE_EAGER))

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    # HuggingFace auth + accelerated download
    if hf_token := (cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")):
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Point every HF client at our shared cache so weights land in models/
    # alongside the llama_nemotron download, not in ~/.cache/huggingface.
    os.environ["HF_HOME"] = str(model_cache)

    # Blackwell FP4 MoE kernels via FlashInfer — required to unlock the
    # NVFP4 quantization's efficiency; on non-Blackwell hardware vLLM will
    # either fall back silently (emulated FP4, slower) or refuse to start
    # depending on the FlashInfer build.
    os.environ["VLLM_USE_FLASHINFER_MOE_FP4"] = "1"
    os.environ["VLLM_FLASHINFER_MOE_BACKEND"] = "throughput"

    parser_path = _ensure_reasoning_parser(model_cache)

    argv = [
        "vllm", "serve", model,
        "--served-model-name", "llm",
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser-plugin", str(parser_path),
        "--reasoning-parser", "nano_v3",
        "--kv-cache-dtype", "fp8",
        "--max-num-seqs", str(max_num_seqs),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    if enforce_eager:
        argv.append("--enforce-eager")

    print(
        f"[nemotron3_nano_llm_server] Launching vLLM on "
        f"http://{host}:{port}/v1  model={model}",
        flush=True,
    )
    print(f"[nemotron3_nano_llm_server] argv={' '.join(argv)}", flush=True)

    # execvp replaces this Python process with vllm so the launcher's
    # signal-forwarding goes straight to the inference server.
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    run()
