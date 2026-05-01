#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Produce ../web-xr/vendor/{cloudxr-sdk,livekit-client}.esm.mjs so the
# XR render demo page loads both same-origin (works on headsets / offline LANs).
# Output files are gitignored — run this once on the host that serves the demo.
#
#   cloudxr-sdk.esm.mjs    — webpack-bundled from @nvidia/cloudxr (NGC tarball)
#   livekit-client.esm.mjs — copied from npm's prebuilt ESM

set -Eeuo pipefail

cd "$(dirname "$0")"

VERSION="$(tr -d '[:space:]' < .sdk-version)"
SDK_FILE="nvidia-cloudxr-${VERSION}.tgz"
LOCAL_TARBALL="sdk.tgz"
VENDOR_DIR="../web-xr/vendor"
OUT_CLOUDXR="${VENDOR_DIR}/cloudxr-sdk.esm.mjs"
OUT_LIVEKIT="${VENDOR_DIR}/livekit-client.esm.mjs"

mkdir -p "${VENDOR_DIR}"

echo "CloudXR SDK version: ${VERSION}"

# ── 1. Obtain sdk.tgz ────────────────────────────────────────────────────────
# Download (or copy) into a sibling .partial file first and rename only on
# success — otherwise an interrupted curl/wget / failed cp leaves a truncated
# sdk.tgz in place that future runs happily reuse, producing a corrupt bundle
# whose root cause looks like "npm install crashed in @nvidia/cloudxr".
#
# We don't pin a checksum here because the SDK tarball is fetched live from
# NGC and only the .sdk-version pin is committed; the npm install + webpack
# build will fail loudly on a corrupt tarball, so this only protects against
# the silent "previous run was interrupted" failure mode.
if [ -f "${LOCAL_TARBALL}" ]; then
    echo "Using existing ${LOCAL_TARBALL}"
else
    PARTIAL="${LOCAL_TARBALL}.partial"
    rm -f "${PARTIAL}"
    # Convenience fallback: if IsaacTeleop has already downloaded the same
    # version, reuse it instead of hitting the network.
    ISAAC_COPY="${HOME}/hub/IsaacTeleop/deps/cloudxr/${SDK_FILE}"
    if [ -f "${ISAAC_COPY}" ]; then
        echo "Copying from IsaacTeleop: ${ISAAC_COPY}"
        cp "${ISAAC_COPY}" "${PARTIAL}"
    else
        # Public NGC resource API; no NGC CLI required.
        NGC_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/cloudxr-js/${VERSION}/files?redirect=true&path=${SDK_FILE}"
        echo "Downloading from public NGC:"
        echo "  ${NGC_URL}"
        if command -v wget >/dev/null 2>&1; then
            wget --content-disposition --output-document "${PARTIAL}" "${NGC_URL}"
        elif command -v curl >/dev/null 2>&1; then
            curl -fL --output "${PARTIAL}" "${NGC_URL}"
        else
            echo "error: need wget or curl to download the SDK" >&2
            exit 1
        fi
    fi
    mv "${PARTIAL}" "${LOCAL_TARBALL}"
fi

# ── 2. Install deps ──────────────────────────────────────────────────────────
echo "Running npm install…"
# --legacy-peer-deps: @nvidia/cloudxr declares peerDependencies on gl-matrix
# and long. We satisfy them in our own dependencies list, but newer npm will
# still raise ERESOLVE when the peer's own tarball appears twice in the tree.
npm install --no-audit --no-fund --legacy-peer-deps

# ── 3. Bundle CloudXR to ESM ─────────────────────────────────────────────────
echo "Bundling CloudXR…"
npm run build

if [ ! -f "${OUT_CLOUDXR}" ]; then
    echo "error: expected ${OUT_CLOUDXR} was not produced" >&2
    exit 1
fi

# ── 4. Copy the prebuilt livekit-client ESM ──────────────────────────────────
LIVEKIT_SRC="node_modules/livekit-client/dist/livekit-client.esm.mjs"
if [ ! -f "${LIVEKIT_SRC}" ]; then
    echo "error: ${LIVEKIT_SRC} not found after npm install" >&2
    exit 1
fi
cp "${LIVEKIT_SRC}" "${OUT_LIVEKIT}"
echo "Copied $(basename "${OUT_LIVEKIT}")  ($(stat -c%s "${OUT_LIVEKIT}") bytes)"

echo
echo "Done. Vendor bundles ready under ${VENDOR_DIR}/."
