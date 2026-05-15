#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify the `kimera_vio` docker image is healthy without needing an
# external dataset download.  Kimera-VIO ships a 95-frame stereo + IMU
# `MicroEurocDataset` in its source tree at tests/data/; this script
# pulls the source into a local checkout (if needed), runs Kimera on
# the micro dataset inside the container, and computes ATE against
# the bundled ground truth.
#
# On a 4-core CPU box the full run takes ~2-3 seconds.  Expected ATE-
# RMSE on the micro dataset is well under 1 cm.

set -euo pipefail

CHECKOUT="${CHECKOUT:-/tmp/Kimera-VIO}"
IMAGE="${IMAGE:-kimera_vio}"
OUT="${OUT:-/tmp/kimera_verify_out}"

if [[ ! -d "$CHECKOUT/tests/data/MicroEurocDataset" ]]; then
  echo "[verify] cloning Kimera-VIO to $CHECKOUT (need the bundled micro dataset)"
  git clone --depth 1 https://github.com/MIT-SPARK/Kimera-VIO.git "$CHECKOUT"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[verify] docker image '$IMAGE' not found — build it first:" >&2
  echo "    docker build -f $CHECKOUT/Dockerfile_20_04 -t kimera_vio_deps $CHECKOUT" >&2
  echo "    docker build -f $(dirname "$0")/Dockerfile.kimera -t $IMAGE ." >&2
  exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"
echo "[verify] running Kimera-VIO on $CHECKOUT/tests/data/MicroEurocDataset"
docker run --rm \
    -v "$CHECKOUT/tests/data/MicroEurocDataset":/dataset:ro \
    -v "$OUT":/output:rw \
    "$IMAGE" \
    bash -lc "cd /root/Kimera-VIO/build && ./stereoVIOEuroc \
        --dataset_type=0 \
        --dataset_path=/dataset \
        --params_folder_path=/root/Kimera-VIO/params/Euroc \
        --log_output=true \
        --output_path=/output \
        --initial_k=0 --final_k=95 \
        --visualize=false --use_lcd=false \
        --vocabulary_path=/root/Kimera-VIO/vocabulary/ORBvoc.yml \
        --logtostderr=1 --colorlogtostderr=0" 2>&1 | tail -5

cp "$CHECKOUT/tests/data/MicroEurocDataset/mav0/state_groundtruth_estimate0/data.csv" "$OUT/gt.csv"

echo
echo "[verify] ATE vs bundled ground truth:"
python3 - "$OUT/traj_vio.csv" "$OUT/gt.csv" <<'PY' || true
import sys
try:
    import numpy as np
except ImportError:
    print("    (numpy not installed in host python — install or run inside a venv)")
    sys.exit(0)

est_path, gt_path = sys.argv[1], sys.argv[2]
def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = line.split(",")
        if not p[0].lstrip("-").replace(".","",1).isdigit(): continue
        rows.append((float(p[0])/1e9, np.array([float(p[1]), float(p[2]), float(p[3])])))
    return rows
est = load(est_path); gt = load(gt_path)
gt_ts = np.array([t for t,_ in gt])
pairs = []
for ts, te in est:
    j = int(np.argmin(np.abs(gt_ts - ts)))
    if abs(gt_ts[j] - ts) <= 0.02:
        pairs.append((te, gt[j][1]))
print(f"    paired {len(pairs)} / est={len(est)} gt={len(gt)}")
if len(pairs) >= 4:
    Et = np.stack([p[0] for p in pairs]); Gt = np.stack([p[1] for p in pairs])
    N = len(Et)
    mu_e, mu_g = Et.mean(0), Gt.mean(0)
    Ec, Gc = Et - mu_e, Gt - mu_g
    Sigma = (Ec.T @ Gc) / N
    U, S, Vt = np.linalg.svd(Sigma)
    D = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: D[2,2] = -1
    var_e = (Ec**2).sum() / N
    scale = float((S * np.diag(D)).sum() / var_e) if var_e > 1e-12 else 1.0
    R = U @ D @ Vt
    t = mu_g - scale * R @ mu_e
    aligned = (scale * (R @ Et.T)).T + t
    errs = np.linalg.norm(aligned - Gt, axis=1)
    print(f"    scale: {scale:.3f}")
    print(f"    ATE-RMSE: {np.sqrt((errs**2).mean())*100:.2f} cm")
    print(f"    mean:     {errs.mean()*100:.2f} cm")
    print(f"    max:      {errs.max()*100:.2f} cm")
PY
echo
echo "[verify] output left in $OUT/  (traj_vio.csv, frontend_images/, …)"
