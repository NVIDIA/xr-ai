#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install DPVO (Deep Patch Visual Odometry, MIT license) and its CUDA
# dependencies into the mono-slam-example worker's uv-managed venv.
#
# Requirements:
#   - CUDA 12.1 or later (nvcc must be on PATH)
#   - Python 3.11 or 3.12
#   - A GPU (DPVO does not run on CPU)
#   - `uv` on PATH (used to materialize the worker venv if missing)
#
# Usage (from anywhere):
#   bash scripts/install_dpvo.sh
#
# The script always targets agent-samples/mono-slam-example/worker/.venv —
# that is the venv `uv run mono_slam_example` ultimately spawns the worker in,
# so DPVO + torch must land there. Running plain `pip install` would put them
# in system Python or the orchestrator venv and the worker would still fail
# with `ModuleNotFoundError: torch`.
#
# After installation:
#   The dpvo package is available to the worker as "import dpvo".
#   Download model weights with: bash scripts/download_dataset.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPVO_DIR="${REPO_ROOT}/deps/dpvo"
EIGEN_VERSION="3.4.0"
EIGEN_ZIP="eigen-${EIGEN_VERSION}.zip"
EIGEN_SHA256="8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
TORCH_VERSION="2.3.1+cu121"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
TORCH_SCATTER_INDEX="https://data.pyg.org/whl/torch-2.3.1+cu121.html"

# ── 0. Resolve the worker venv ─────────────────────────────────────────────────
WORKER_DIR="${REPO_ROOT}/agent-samples/mono-slam-example/worker"
WORKER_VENV="${WORKER_DIR}/.venv"

if [[ ! -d "${WORKER_VENV}" ]]; then
    if ! command -v uv &>/dev/null; then
        echo "ERROR: worker venv missing at ${WORKER_VENV} and 'uv' is not on PATH." >&2
        echo "       Install uv (https://docs.astral.sh/uv/) or run 'uv sync' in" >&2
        echo "       ${WORKER_DIR} to create the venv first." >&2
        exit 1
    fi
    echo "Worker venv not found — creating it via 'uv sync' ..."
    (cd "${WORKER_DIR}" && uv sync --quiet)
fi

PIP="${WORKER_VENV}/bin/pip"
PYBIN="${WORKER_VENV}/bin/python"
if [[ ! -x "${PIP}" ]] || [[ ! -x "${PYBIN}" ]]; then
    echo "ERROR: ${WORKER_VENV} exists but is missing pip/python." >&2
    exit 1
fi

echo "=== DPVO installer ==="
echo "Target venv: ${WORKER_VENV}"
echo "DPVO dir:    ${DPVO_DIR}"
echo "Torch:       ${TORCH_VERSION}"

# ── 1. Check nvcc ───────────────────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    # Try the common CUDA toolkit location before failing.
    for cuda_path in /usr/local/cuda/bin /usr/local/cuda-12.6/bin /usr/local/cuda-12.1/bin; do
        if [[ -x "${cuda_path}/nvcc" ]]; then
            export PATH="${cuda_path}:${PATH}"
            echo "Found nvcc at ${cuda_path}"
            break
        fi
    done
fi
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit (>= 12.1) and add it to PATH." >&2
    exit 1
fi
echo "nvcc: $(nvcc --version | head -1)"

# ── 2. Install torch 2.3.1+cu121 ───────────────────────────────────────────────
echo "Installing torch ${TORCH_VERSION} ..."
"${PIP}" install \
    "torch==${TORCH_VERSION}" \
    "torchvision>=0.18.1+cu121" \
    --index-url "${TORCH_INDEX}" \
    --no-deps --quiet

# ── 3. Install torch-scatter (MIT-licensed, PyG wheel) ─────────────────────────
echo "Installing torch-scatter ..."
"${PIP}" install "torch-scatter==2.1.2" \
    --find-links "${TORCH_SCATTER_INDEX}" \
    --no-deps --quiet

# ── 4. Install DPVO Python dependencies ────────────────────────────────────────
echo "Installing DPVO Python dependencies ..."
"${PIP}" install \
    "scipy>=1.11" \
    "pypose>=0.6.7" \
    "einops>=0.7" \
    "numba>=0.59" \
    "yacs>=0.1.8" \
    "matplotlib>=3.7" \
    "opencv-python-headless>=4.8" \
    --quiet

# ── 5. Clone DPVO if not already present ───────────────────────────────────────
if [[ ! -d "${DPVO_DIR}/.git" ]]; then
    echo "Cloning DPVO ..."
    mkdir -p "$(dirname "${DPVO_DIR}")"
    git clone --depth 1 https://github.com/princeton-vl/DPVO.git "${DPVO_DIR}"
else
    echo "DPVO already cloned at ${DPVO_DIR}"
fi

# ── 6. Download and verify Eigen 3.4.0 ─────────────────────────────────────────
EIGEN_DIR="${DPVO_DIR}/thirdparty/eigen-${EIGEN_VERSION}"
if [[ ! -d "${EIGEN_DIR}" ]]; then
    echo "Downloading Eigen ${EIGEN_VERSION} ..."
    TMP_ZIP="$(mktemp --suffix=.zip)"
    wget -q "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/${EIGEN_ZIP}" \
        -O "${TMP_ZIP}"

    # Verify checksum — pin against supply-chain tampering.
    ACTUAL_SHA="$(sha256sum "${TMP_ZIP}" | awk '{print $1}')"
    if [[ "${ACTUAL_SHA}" != "${EIGEN_SHA256}" ]]; then
        echo "ERROR: Eigen SHA256 mismatch." >&2
        echo "  expected: ${EIGEN_SHA256}" >&2
        echo "  got:      ${ACTUAL_SHA}" >&2
        rm -f "${TMP_ZIP}"
        exit 1
    fi
    echo "Eigen checksum OK"

    mkdir -p "${DPVO_DIR}/thirdparty"
    unzip -q "${TMP_ZIP}" -d "${DPVO_DIR}/thirdparty"
    # The zip unpacks as eigen-3.4.0/ — rename if needed to match setup.py expectation.
    if [[ ! -d "${EIGEN_DIR}" ]]; then
        # Zip may unpack as eigen-3.4.0 directly — check.
        EXTRACTED="$(find "${DPVO_DIR}/thirdparty" -maxdepth 1 -name 'eigen-*' -type d | head -1)"
        if [[ -n "${EXTRACTED}" && "${EXTRACTED}" != "${EIGEN_DIR}" ]]; then
            mv "${EXTRACTED}" "${EIGEN_DIR}"
        fi
    fi
    rm -f "${TMP_ZIP}"
    echo "Eigen extracted to ${EIGEN_DIR}"
else
    echo "Eigen already present at ${EIGEN_DIR}"
fi

# ── 7. Build and install DPVO CUDA extensions ──────────────────────────────────
echo "Building DPVO CUDA extensions (this takes a few minutes) ..."
cd "${DPVO_DIR}"
PATH="$(dirname "$(command -v nvcc)"):${PATH}" \
"${PIP}" install --no-build-isolation -e . --quiet

echo ""
echo "=== DPVO installation complete ==="
echo "Verify with:  ${PYBIN} -c 'import dpvo; print(dpvo.__file__)'"
echo ""
echo "Download model weights:"
echo "  wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip"
echo "  unzip models.zip -d ${REPO_ROOT}/models"
