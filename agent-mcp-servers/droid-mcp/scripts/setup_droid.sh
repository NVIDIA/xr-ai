#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install DROID-SLAM into the droid-mcp venv.
#
# DROID isn't pip-installable: it ships custom CUDA extensions
# (correlation sampler + DBA layer) that need `python setup.py install`
# linked against the same PyTorch the runtime uses.  Run this script
# once inside the activated droid-mcp venv:
#
#     cd agent-mcp-servers/droid-mcp
#     uv sync                               # installs torch
#     uv run bash scripts/setup_droid.sh
#
# Requirements:
#   * NVIDIA driver + CUDA toolkit (matching the torch wheel's CUDA)
#   * g++ + ninja-build
#   * git
#   * curl
#
# After this script runs, `droid_mcp_server` can be started normally.

set -euo pipefail

CACHE_DIR="${XR_AI_CACHE_DIR:-$HOME/.cache/xr-ai}"
DROID_SRC="$CACHE_DIR/DROID-SLAM"
WEIGHTS_PATH="$CACHE_DIR/droid.pth"

# Upstream pretrained model URL (published by Teed & Deng on Google
# Drive — host yourself or rehost behind a stable mirror for CI use).
WEIGHTS_URL="${DROID_WEIGHTS_URL:-https://drive.google.com/uc?export=download&id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh}"

mkdir -p "$CACHE_DIR"

echo "[setup-droid] cache dir: $CACHE_DIR"

# 1. Clone DROID-SLAM source.
if [[ ! -d "$DROID_SRC/.git" ]]; then
    echo "[setup-droid] cloning DROID-SLAM …"
    git clone --depth 1 https://github.com/princeton-vl/DROID-SLAM.git "$DROID_SRC"
else
    echo "[setup-droid] DROID-SLAM already cloned at $DROID_SRC"
fi

# 2. Build the CUDA extensions + install as an editable package so
#    `import droid_slam` works inside the droid-mcp venv.
echo "[setup-droid] building CUDA extensions …"
( cd "$DROID_SRC" && python setup.py install )

# 3. Fetch the pretrained checkpoint.
if [[ -f "$WEIGHTS_PATH" ]]; then
    echo "[setup-droid] weights already at $WEIGHTS_PATH (skipping download)"
else
    echo "[setup-droid] downloading weights → $WEIGHTS_PATH"
    if command -v gdown >/dev/null 2>&1; then
        gdown -O "$WEIGHTS_PATH" "$WEIGHTS_URL"
    else
        echo "[setup-droid] gdown not installed; falling back to curl. "
        echo "[setup-droid] If this fails, pip install gdown and re-run."
        curl -L -o "$WEIGHTS_PATH" "$WEIGHTS_URL"
    fi
fi

echo "[setup-droid] done.  Weights: $WEIGHTS_PATH"
echo "[setup-droid] Next: uv run python -m droid_mcp_server --config droid_mcp_server.yaml"
