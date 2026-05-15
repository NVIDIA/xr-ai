#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compute Absolute Trajectory Error of Kimera's trajectory output
# against the EuRoC ground truth that ships with each sequence.
#
# Uses the same Umeyama Sim(3)-alignment as pose-mcp's TUM bench so
# the numbers are directly comparable.

set -euo pipefail
DEST="${DEST:-/tmp/euroc}"
SEQ="${SEQ:-V1_01_easy}"

DATASET="$DEST/$SEQ"
EST="$DATASET/output_logs/traj_vio.csv"
GT="$DATASET/mav0/state_groundtruth_estimate0/data.csv"

if [[ ! -f "$EST" ]]; then echo "[eval] no estimate at $EST — run run_euroc.sh first" >&2; exit 1; fi
if [[ ! -f "$GT"  ]]; then echo "[eval] no ground truth at $GT" >&2; exit 1; fi

python3 - "$EST" "$GT" <<'PY'
import csv, sys
import numpy as np

est_path, gt_path = sys.argv[1], sys.argv[2]

def load_traj(path, ts_col, t_cols):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(',') if ',' in line else line.split()
            if not parts[0].lstrip('-').replace('.', '', 1).isdigit():
                continue
            ts = float(parts[ts_col]) / (1e9 if len(parts[ts_col].rstrip('.')) > 13 else 1.0)
            t = np.array([float(parts[c]) for c in t_cols])
            rows.append((ts, t))
    return rows

# Kimera traj_vio.csv: # timestamp(ns), x, y, z, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz
est = load_traj(est_path, 0, [1, 2, 3])
# EuRoC GT data.csv: # timestamp, p_RS_R_x [m], p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z, ...
gt  = load_traj(gt_path,  0, [1, 2, 3])

# Pair by nearest timestamp (within 20 ms).
gt_ts = np.array([t for t, _ in gt])
pairs = []
for ts, t_est in est:
    j = int(np.argmin(np.abs(gt_ts - ts)))
    if abs(gt_ts[j] - ts) <= 0.02:
        pairs.append((t_est, gt[j][1]))
if len(pairs) < 4:
    print("[eval] fewer than 4 paired samples — alignment impossible"); sys.exit(2)
est_t = np.stack([p[0] for p in pairs], axis=0)
gt_t  = np.stack([p[1] for p in pairs], axis=0)

# Umeyama Sim(3) alignment (translation + rotation + scale).
N    = len(est_t)
mu_e = est_t.mean(axis=0); mu_g = gt_t.mean(axis=0)
Ec   = est_t - mu_e;       Gc   = gt_t  - mu_g
Sigma_xy = (Ec.T @ Gc) / float(N)
U, S, Vt = np.linalg.svd(Sigma_xy)
D = np.eye(3)
if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[2, 2] = -1
R = U @ D @ Vt
var_e = (Ec ** 2).sum() / float(N)
scale = float((S * np.diag(D)).sum() / var_e) if var_e > 1e-12 else 1.0
t = mu_g - scale * R @ mu_e
aligned = (scale * (R @ est_t.T)).T + t
errs = np.linalg.norm(aligned - gt_t, axis=1)

print(f"== Kimera-VIO ATE on EuRoC =={len(pairs)} paired samples")
print(f"  scale recovered: {scale:.3f}")
print(f"  RMSE: {np.sqrt((errs**2).mean()) * 100:.2f} cm")
print(f"  mean: {errs.mean() * 100:.2f} cm")
print(f"  max : {errs.max()  * 100:.2f} cm")
PY
