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
# Run the version probes with pipefail OFF — `head -1` legitimately
# closes its stdin early, which gives the upstream `sed` SIGPIPE.
# Under `set -euo pipefail` that surfaces as exit 141 and kills the
# whole script before any of the preflight messages can print.
set +o pipefail
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
# Probe torch — capture stderr too so we can show the real error if
# import fails (the common case here is a libnccl ABI mismatch when
# an older system NCCL is ahead of torch/lib on LD_LIBRARY_PATH).
TORCH_PROBE="$(python -c 'import torch, sys; sys.stdout.write(torch.version.cuda or "")' 2>&1)"
set -o pipefail
TORCH_VER="${TORCH_PROBE%%.*}"
# If the probe contains the substring "Error", import failed.
if echo "$TORCH_PROBE" | grep -q -E '(Error|Traceback)' >/dev/null; then
    cat >&2 <<EOF
[setup-droid] ERROR: torch is installed but won't import in this venv.

              Probe output:
              $(echo "$TORCH_PROBE" | sed 's/^/                  /')

              Common cause: an older system NCCL / CUDA on LD_LIBRARY_PATH is
              shadowing the torch wheel's bundled libnccl.so.2.  Look for the
              symbol name in the error above — if it's an ncclCommXxx symbol,
              your system NCCL is older than the one torch needs.

              Fixes:

              (A) Remove the system CUDA libdir from LD_LIBRARY_PATH for this
                  shell (cleanest if torch's bundled libs are self-sufficient):
                      unset LD_LIBRARY_PATH
                      uv run slam_example

              (B) Edit ~/.bashrc to drop the /usr/local/cuda-*/lib64 entry,
                  or switch it to the toolkit version matching torch.
EOF
    exit 2
fi
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

# 0b. GPU compute-capability vs nvcc-max-supported sanity check.  Ada
#     (sm_89, RTX 40-series / L40) and Hopper (sm_90, H100) need
#     nvcc >= 11.8.  Older nvcc dies with "Unsupported gpu architecture
#     'compute_89'" deep in the build, easy to miss.  Catch it here.
if command -v nvidia-smi >/dev/null 2>&1 && [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    set +o pipefail
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)"
    NVCC_MINOR="$(nvcc --version | sed -n 's/.*release [0-9][0-9]*\.\([0-9][0-9]*\).*/\1/p' | head -1)"
    set -o pipefail
    GPU_CC_INT="${GPU_CC//.}"   # "8.9" -> "89"
    if [[ -n "$GPU_CC_INT" && "$NVCC_VER" == "11" ]] \
       && [[ "${NVCC_MINOR:-0}" =~ ^[0-9]+$ ]] \
       && (( NVCC_MINOR < 8 )) \
       && [[ "$GPU_CC_INT" =~ ^[0-9]+$ ]] \
       && (( GPU_CC_INT > 86 )); then
        cat >&2 <<EOF
[setup-droid] WARN: GPU reports sm_${GPU_CC_INT} but nvcc 11.${NVCC_MINOR} only
              supports up to sm_86 (Ampere).  The compile will fail with
              "nvcc fatal: Unsupported gpu architecture 'compute_${GPU_CC_INT}'".

              Pick one fix and re-run \`uv run slam_example\`:

              (A) Constrain TORCH_CUDA_ARCH_LIST so PyTorch asks nvcc for an arch
                  it supports.  The driver JIT-promotes at runtime to your real GPU:
                      export TORCH_CUDA_ARCH_LIST="8.6+PTX"

              (B) Install a newer nvcc (existing 11.5 stays in place):
                      sudo apt install cuda-toolkit-11-8
                      export PATH=/usr/local/cuda-11.8/bin:\$PATH

              Continuing anyway in case you know what you're doing …
EOF
    fi
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
