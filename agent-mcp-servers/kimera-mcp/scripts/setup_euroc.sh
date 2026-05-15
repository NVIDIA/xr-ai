#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download + unzip the EuRoC V1_01_easy sequence — the smallest one that
# still exercises full VIO (camera + IMU + ground truth).  ~700 MB.
# Drop bigger sequences in the same dir to test on harder data.

set -euo pipefail

DEST="${1:-/tmp/euroc}"
SEQ="${2:-V1_01_easy}"
URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/${SEQ}/${SEQ}.zip"

mkdir -p "$DEST"
cd "$DEST"

if [[ -d "$SEQ/mav0" ]]; then
  echo "[setup_euroc] $SEQ already extracted at $DEST/$SEQ"
  exit 0
fi

echo "[setup_euroc] downloading $URL"
if [[ ! -f "${SEQ}.zip" ]]; then
  curl -L -o "${SEQ}.zip" "$URL"
fi

echo "[setup_euroc] unzipping into $DEST/$SEQ"
mkdir -p "$SEQ"
unzip -q -o "${SEQ}.zip" -d "$SEQ"

echo "[setup_euroc] ready: $DEST/$SEQ"
ls "$DEST/$SEQ/mav0/" 2>/dev/null || echo "[setup_euroc] WARNING: extraction layout unexpected"
