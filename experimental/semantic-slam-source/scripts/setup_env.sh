#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# First-launch environment setup for semantic-slam.
#
# Installs EVERYTHING needed to run the real pipeline, using uv for Python
# isolation. Idempotent: each step is guarded and skipped if already done, so
# it is safe to call on every launch. A sentinel file short-circuits the whole
# script once a full setup has succeeded (delete it or pass --force to redo).
#
#   1. core Python deps          -> uv sync            (isolated .venv)
#   2. CUDA build toolchain      -> nvcc + gcc-10/g++-10 (apt, only if missing)
#   3. model stack (CUDA exts)   -> chamferdist, pytorch3d, gradslam,
#                                   segment-anything   (built against venv torch)
#   4. SAM weights               -> $GSA_PATH/sam_vit_h_4b8939.pth
#
# Usage:  scripts/setup_env.sh [--force] [--with-dataset]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
SENTINEL="$VENV/.semantic_slam_setup_complete"
GSA_PATH="${GSA_PATH:-$REPO_ROOT/external/Grounded-Segment-Anything}"
# Ada (L40S, sm_89) is built as 8.6+PTX so it also works with CUDA toolkits
# older than 11.8 (PTX is JIT-compiled at runtime). Override if needed.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

FORCE=0; WITH_DATASET=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --with-dataset) WITH_DATASET=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }

if [[ -f "$SENTINEL" && "$FORCE" -eq 0 ]]; then
  log "already set up ($SENTINEL). Use --force to redo."
  exit 0
fi

command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: https://docs.astral.sh/uv/" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Core Python deps into the isolated venv.
# ---------------------------------------------------------------------------
# --inexact: never remove packages that aren't in the lock. The git-installed
# model stack (pytorch3d/gradslam/...) lives in the venv but not the lockfile,
# so a plain `uv sync` would delete it. --inexact leaves it untouched.
log "uv sync --inexact (core deps)"
uv sync --inexact

PY="$VENV/bin/python"

# Quick check: is the heavy model stack already importable? If so, skip 2-3.
if "$PY" - <<'PY' 2>/dev/null
import importlib
for m in ("chamferdist", "pytorch3d", "gradslam", "segment_anything",
          "groundingdino", "ram"):
    importlib.import_module(m)
PY
then
  log "model stack already installed; skipping CUDA build."
else
  # -------------------------------------------------------------------------
  # 2. CUDA build toolchain (only if a CUDA extension actually needs building).
  #    Needs nvcc + a gcc/g++ that nvcc accepts as host compiler. CUDA 11.5's
  #    nvcc miscompiles gcc-11 libstdc++ headers, so we pin gcc-10/g++-10.
  # -------------------------------------------------------------------------
  need_apt=0
  command -v nvcc >/dev/null 2>&1 || need_apt=1
  command -v gcc-10 >/dev/null 2>&1 || need_apt=1
  command -v g++-10 >/dev/null 2>&1 || need_apt=1
  if [[ "$need_apt" -eq 1 ]]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
      log "installing CUDA toolkit + gcc-10/g++-10 via apt"
      sudo -n apt-get update -qq
      sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        nvidia-cuda-toolkit build-essential gcc-10 g++-10
    else
      warn "nvcc/gcc-10/g++-10 missing and passwordless sudo unavailable."
      warn "Install manually: nvidia-cuda-toolkit gcc-10 g++-10, then re-run."
      exit 1
    fi
  fi

  export CUDA_HOME="${CUDA_HOME:-/usr}"
  export CC="${CC:-$(command -v gcc-10)}"
  export CXX="${CXX:-$(command -v g++-10)}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-8}"

  # -------------------------------------------------------------------------
  # 3. Model stack. torch 2.0.1's cpp_extension needs pkg_resources.packaging
  #    (removed in setuptools>=70), so pin the build-time setuptools. These are
  #    built with --no-build-isolation against the venv's torch.
  # -------------------------------------------------------------------------
  log "installing build helpers (setuptools<70, wheel, ninja)"
  uv pip install "setuptools==69.5.1" wheel ninja

  log "building chamferdist (CUDA ext) — this can take a few minutes"
  uv pip install --no-build-isolation chamferdist

  # --no-deps for the git packages: gradslam pins an ancient open3d and
  # pytorch3d would otherwise re-resolve torch. Their real runtime deps
  # (fvcore, iopath, chamferdist, open3d, kornia, ...) are in the core deps.
  log "building pytorch3d (CUDA ext) — this is the slow one (~15-30 min)"
  uv pip install --no-build-isolation --no-deps \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

  log "installing gradslam + segment-anything + groundingdino + recognize-anything (git, --no-deps)"
  # slam/models/__init__.py eagerly imports detection (GroundingDINO),
  # segmentation (SAM), clip and captioning (RAM), so all four are required to
  # import the package even in SAM-only mode. groundingdino builds a CUDA ext.
  uv pip install --no-build-isolation --no-deps \
    "gradslam @ git+https://github.com/gradslam/gradslam.git@conceptfusion" \
    "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git" \
    "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git" \
    "ram @ git+https://github.com/xinyu1205/recognize-anything.git"

  "$PY" - <<'PY'
import chamferdist, pytorch3d, pytorch3d.ops, gradslam, segment_anything  # noqa
import groundingdino.util.inference, ram  # noqa
print("[setup] model stack imports OK")
PY
fi

# ---------------------------------------------------------------------------
# 3b. Generate gRPC protobuf stubs (vis_pb2 / xr_service_pb2). These are
#     generated, not committed; the pipeline imports them at load time.
# ---------------------------------------------------------------------------
if [[ ! -f "$REPO_ROOT/slam/protocols/vis_proto/vis_pb2.py" ]]; then
  log "generating vis_proto stubs"
  ( cd "$REPO_ROOT/slam/protocols/vis_proto" && \
    "$PY" -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vis.proto )
fi
if [[ ! -f "$REPO_ROOT/server/xr_service_pb2.py" ]]; then
  log "generating xr_service stubs"
  ( cd "$REPO_ROOT/server" && \
    "$PY" -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. xr_service.proto )
fi

# ---------------------------------------------------------------------------
# 4. SAM weights at the path the config expects ($GSA_PATH/sam_vit_h_4b8939.pth).
# ---------------------------------------------------------------------------
mkdir -p "$GSA_PATH"
SAM_CKPT="$GSA_PATH/sam_vit_h_4b8939.pth"
if [[ ! -s "$SAM_CKPT" ]]; then
  log "downloading SAM ViT-H weights -> $SAM_CKPT (~2.4GB)"
  curl -fL --retry 3 -o "$SAM_CKPT" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
  log "SAM weights present."
fi

# ---------------------------------------------------------------------------
# Optional: Replica dataset for the end-to-end test.
# ---------------------------------------------------------------------------
if [[ "$WITH_DATASET" -eq 1 ]]; then
  REPLICA_ROOT="${REPLICA_ROOT:-/data/replica/Replica}"
  if [[ ! -d "$REPLICA_ROOT/room0" ]]; then
    log "downloading Replica dataset -> $(dirname "$REPLICA_ROOT") (~12GB)"
    mkdir -p "$(dirname "$REPLICA_ROOT")"
    curl -fL --retry 3 -o "$(dirname "$REPLICA_ROOT")/Replica.zip" \
      https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
    ( cd "$(dirname "$REPLICA_ROOT")" && unzip -q -o Replica.zip )
  else
    log "Replica dataset present at $REPLICA_ROOT."
  fi
fi

touch "$SENTINEL"
log "setup complete. GSA_PATH=$GSA_PATH"
