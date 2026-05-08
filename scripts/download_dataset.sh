#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the TUM RGB-D fr1/xyz sequence for DPVO benchmark evaluation.
#
# Dataset: TUM RGB-D Benchmark (https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
# Sequence: freiburg1_xyz (~47 MB) — shortest TUM sequence; DPVO supports it
#           with hard-coded intrinsics (freiburg1 camera).
# License:  Creative Commons Attribution 4.0 (CC-BY-4.0) — compatible with
#           the project's Apache-2.0 codebase.
#
# Usage:
#   bash scripts/download_dataset.sh [OUTPUT_DIR]
#
#   OUTPUT_DIR defaults to datasets/tum inside the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${REPO_ROOT}/datasets/tum}"

TUM_BASE="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
SEQUENCE="rgbd_dataset_freiburg1_xyz"
TARBALL="${SEQUENCE}.tgz"

mkdir -p "${OUTPUT_DIR}"

echo "=== TUM RGB-D download ==="
echo "Sequence:   ${SEQUENCE}"
echo "Output dir: ${OUTPUT_DIR}"

if [[ -d "${OUTPUT_DIR}/${SEQUENCE}" ]]; then
    echo "Sequence already present at ${OUTPUT_DIR}/${SEQUENCE} — skipping download."
    exit 0
fi

TMP_TAR="${OUTPUT_DIR}/${TARBALL}"
echo "Downloading ${TARBALL} ..."
wget -q --show-progress \
    "${TUM_BASE}/${TARBALL}" \
    -O "${TMP_TAR}"

echo "Extracting ..."
tar -xzf "${TMP_TAR}" -C "${OUTPUT_DIR}"
rm -f "${TMP_TAR}"

echo ""
echo "=== Download complete ==="
echo "Sequence at: ${OUTPUT_DIR}/${SEQUENCE}"
echo ""
echo "Run benchmark:"
echo "  pytest tests/test_mono_slam_benchmark.py -v"
echo "     --dataset-dir ${OUTPUT_DIR}/${SEQUENCE}"
echo "     --weights-path models/dpvo.pth"
