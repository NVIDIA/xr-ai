#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Kimera-VIO inside the `kimera_vio` docker image on a downloaded
# EuRoC sequence.  Produces a trajectory CSV under
# /tmp/euroc/<seq>/output_logs/ that the eval script can compare against
# the ground truth file Kimera ships with each EuRoC sequence.
#
# Defaults to monocular mode (params/EurocMono).  Pass MODE=stereo for
# the standard stereo VIO pipeline.

set -euo pipefail

DEST="${DEST:-/tmp/euroc}"
SEQ="${SEQ:-V1_01_easy}"
MODE="${MODE:-mono}"     # mono | stereo
IMAGE="${IMAGE:-kimera_vio}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[run_euroc] docker image '$IMAGE' not found." >&2
  echo "  build it first:" >&2
  echo "    cd /tmp/Kimera-VIO && docker build -f Dockerfile_20_04 -t $IMAGE ." >&2
  exit 1
fi

DATASET="$DEST/$SEQ"
if [[ ! -d "$DATASET/mav0" ]]; then
  echo "[run_euroc] dataset not found at $DATASET — run setup_euroc.sh first" >&2
  exit 1
fi

# Kimera ships pre-tuned params/Euroc[Mono] inside the image at
# /root/Kimera-VIO/params/.  Pick the right one for the requested mode.
if [[ "$MODE" == "mono" ]]; then
  PARAMS=/root/Kimera-VIO/params/EurocMono
  BINARY=stereoVIOEuroc        # name is shared; --use_stereo flag selects mode
  STEREO_FLAG=--use_stereo=false
else
  PARAMS=/root/Kimera-VIO/params/Euroc
  BINARY=stereoVIOEuroc
  STEREO_FLAG=--use_stereo=true
fi

mkdir -p "$DATASET/output_logs"

echo "[run_euroc] running Kimera-VIO ($MODE) on $SEQ ..."
docker run --rm \
    -v "$DATASET":/dataset:rw \
    "$IMAGE" \
    bash -lc "cd /root/Kimera-VIO/build && \
        ./$BINARY \
          --dataset_type=0 \
          --dataset_path=/dataset \
          --params_folder_path=$PARAMS \
          $STEREO_FLAG \
          --log_output=true \
          --output_path=/dataset/output_logs \
          --initial_k=50 \
          --final_k=2000 \
          --visualize=false \
          --use_lcd=true \
          --vocabulary_path=/root/Kimera-VIO/vocabulary/ORBvoc.yml"

echo "[run_euroc] done — trajectory at $DATASET/output_logs/traj_vio.csv"
