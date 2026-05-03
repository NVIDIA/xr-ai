# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vlm_server_docker — Docker Model Runner launcher for Cosmos-Reason1-7B.

Drop-in replacement for vlm_server (transformers in-process) that uses
Docker Model Runner instead.  Pulls the model then execs into
``docker model run``.

API endpoint (Docker Model Runner):
    http://localhost:12434/engines/llama.cpp/v1

Use this model name in OpenAI API requests:
    hf.co/nvidia/Cosmos-Reason1-7B  (or whatever model: is set to below)

Config keys (vlm_server_docker.yaml)
--------------------------------------
    model: str   Docker Model Runner image (default: hf.co/nvidia/Cosmos-Reason1-7B).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

_DEFAULT_MODEL = "hf.co/nvidia/Cosmos-Reason1-7B"


def _require_docker_model_runner() -> None:
    """Exit with a clear message if the Docker Model Runner plugin is not installed."""
    result = subprocess.run(
        ["docker", "model", "version"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(
            "[vlm_server_docker] ERROR: Docker Model Runner plugin not found.\n"
            "\n"
            "Install it on Linux:\n"
            "  sudo apt-get install docker-model-plugin\n"
            "\n"
            "Or follow the official guide:\n"
            "  https://docs.docker.com/model-runner/\n"
            "\n"
            "Alternatively, use the in-process VLM server:\n"
            "  ai-services/vlm-server/",
            flush=True,
        )
        sys.exit(1)


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

    _require_docker_model_runner()

    model = cfg.get("model", _DEFAULT_MODEL)

    print(f"[vlm_server_docker] Pulling {model}…", flush=True)
    subprocess.run(["docker", "model", "pull", model], check=True)

    print(
        "[vlm_server_docker] Starting Docker Model Runner — "
        "API: http://localhost:12434/engines/llama.cpp/v1",
        flush=True,
    )
    os.execvp("docker", ["docker", "model", "run", model])


if __name__ == "__main__":
    run()
