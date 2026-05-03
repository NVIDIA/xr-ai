# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nemotron3_nano_docker_llm_server — Docker Model Runner launcher for the
Nemotron-3-Nano tool-calling LLM.

Drop-in replacement for nemotron3_nano_llm_server that uses Docker Model
Runner instead of vLLM.  Selects the model variant based on GPU compute
capability, pulls it, then execs into ``docker model run``.

API endpoint (Docker Model Runner):
    http://localhost:12434/engines/llama.cpp/v1

Use this model name in OpenAI API requests (the selected variant):
    hf.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8         (Ada/Hopper)
    hf.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4       (Blackwell)
    hf.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8             (use_small)

Config keys (nemotron3_nano_docker_llm_server.yaml)
----------------------------------------------------
    model_blackwell: str   Model image for Blackwell GPUs (SM100+).
    model_ada:       str   Model image for Ada / Hopper / Ampere.
    model_small:     str   9B variant — low VRAM or development use.
    use_small:       bool  Always use model_small regardless of GPU (default: false).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

_MODEL_BLACKWELL = "hf.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
_MODEL_ADA       = "hf.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
_MODEL_SMALL     = "hf.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"


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


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    if cfg.get("use_small", False):
        model = cfg.get("model_small", _MODEL_SMALL)
        print(f"[nemotron3_nano_docker] use_small=true → {model}", flush=True)
    else:
        major = _gpu_compute_major()
        if major >= 10:
            model = cfg.get("model_blackwell", _MODEL_BLACKWELL)
            print(f"[nemotron3_nano_docker] Blackwell (SM{major}0) → {model}", flush=True)
        else:
            model = cfg.get("model_ada", _MODEL_ADA)
            arch  = f"SM{major}0" if major > 0 else "unknown GPU"
            print(f"[nemotron3_nano_docker] Pre-Blackwell ({arch}) → {model}", flush=True)

    print(f"[nemotron3_nano_docker] Pulling {model}…", flush=True)
    subprocess.run(["docker", "model", "pull", model], check=True)

    print(
        "[nemotron3_nano_docker] Starting Docker Model Runner — "
        "API: http://localhost:12434/engines/llama.cpp/v1",
        flush=True,
    )
    os.execvp("docker", ["docker", "model", "run", model])


if __name__ == "__main__":
    run()
