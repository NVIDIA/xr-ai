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

# 0. Preflight: nvcc on PATH and torch CUDA major-version match.
#    PyTorch's build_extensions() refuses to compile when the major
#    version of `nvcc --version` doesn't match `torch.version.cuda`,
#    and the error it surfaces (one line buried in a 30-line traceback)
#    is easy to miss.  Catch it here with a clear, actionable message.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[setup-droid] ERROR: nvcc not on PATH — install a CUDA toolkit (apt install cuda-toolkit-12-* or download from developer.nvidia.com) and re-run."
    exit 2
fi
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
TORCH_VER="$(python -c 'import torch, sys; v = torch.version.cuda or ""; sys.stdout.write(v.split(".")[0])' 2>/dev/null)"
if [[ -z "$TORCH_VER" || -z "$NVCC_VER" ]]; then
    echo "[setup-droid] WARN: could not detect CUDA versions (nvcc=$NVCC_VER torch=$TORCH_VER); continuing anyway."
elif [[ "$NVCC_VER" != "$TORCH_VER" ]]; then
    cat >&2 <<EOF
[setup-droid] ERROR: CUDA major-version mismatch.
                nvcc  reports CUDA $NVCC_VER.x  (\`$(command -v nvcc)\`)
                torch reports CUDA $TORCH_VER.x ($(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null))
              PyTorch refuses to compile CUDA extensions across major versions.

              Pick one fix:

              (A) Self-contained — re-install torch with a CUDA wheel that matches nvcc:
                    cd agent-mcp-servers/droid-mcp
                    uv pip install --reinstall \\
                        --index-url https://download.pytorch.org/whl/cu${NVCC_VER}8 \\
                        'torch>=2.1,<2.5' 'torchvision>=0.16,<0.20'
                  (cu118 / cu121 / cu124 etc.; pick the one closest to your nvcc.
                   torch >=2.5 dropped CUDA 11 wheels, hence the <2.5 cap when nvcc is 11.x.)

              (B) System — install a CUDA toolkit matching the torch wheel
                  (CUDA $TORCH_VER.x) and put its bin/ first on PATH.
                  Driver permitting, multiple CUDA toolkits coexist fine.

              Then \`rm -rf $DROID_SRC\` to force a clean recompile, and
              re-run \`uv run slam_example\`.
EOF
    exit 2
fi

# 1. Clone DROID-SLAM source.
if [[ ! -d "$DROID_SRC/.git" ]]; then
    echo "[setup-droid] cloning DROID-SLAM …"
    git clone --depth 1 https://github.com/princeton-vl/DROID-SLAM.git "$DROID_SRC"
else
    echo "[setup-droid] DROID-SLAM already cloned at $DROID_SRC"
fi

# 2. Build the CUDA extensions + install as an editable package so
#    `import droid_slam` works inside the droid-mcp venv.
echo "[setup-droid] building CUDA extensions (nvcc=$NVCC_VER, torch CUDA=$TORCH_VER) …"
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
