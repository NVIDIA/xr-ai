#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Produce ../web-xr/vendor/{cloudxr-sdk,livekit-client}.esm.mjs so the
# XR render demo page loads both same-origin (works on headsets / offline LANs).
# Output files are gitignored — run this once on the host that serves the demo.
#
#   cloudxr-sdk.esm.mjs    — webpack-bundled from npm's @nvidia/cloudxr
#   livekit-client.esm.mjs — copied from npm's prebuilt ESM

set -Eeuo pipefail

cd "$(dirname "$0")"

VERSION="$(tr -d '[:space:]' < .sdk-version)"
PACKAGE_VERSION="$(node -p "require('./package.json').dependencies['@nvidia/cloudxr']")"
LIVEKIT_VERSION="$(node -p "require('./package.json').dependencies['livekit-client']")"
VENDOR_DIR="../web-xr/vendor"
OUT_CLOUDXR="${VENDOR_DIR}/cloudxr-sdk.esm.mjs"
OUT_LIVEKIT="${VENDOR_DIR}/livekit-client.esm.mjs"
VERSION_MARKER="${VENDOR_DIR}/.cloudxr-sdk-version"
LIVEKIT_VERSION_MARKER="${VENDOR_DIR}/.livekit-client-version"

if [[ "${PACKAGE_VERSION}" != "${VERSION}" ]]; then
    echo "error: @nvidia/cloudxr must be pinned to .sdk-version (${VERSION}) in package.json" >&2
    exit 1
fi

mkdir -p "${VENDOR_DIR}"
echo "CloudXR SDK version: ${VERSION}"

# ── 1. Install deps ──────────────────────────────────────────────────────────
echo "Running npm install…"
npm install --no-audit --no-fund

# ── 2. Bundle CloudXR to ESM ─────────────────────────────────────────────────
echo "Bundling CloudXR…"
npm run build

if [[ ! -f "${OUT_CLOUDXR}" ]]; then
    echo "error: expected ${OUT_CLOUDXR} was not produced" >&2
    exit 1
fi

# ── 3. Copy the prebuilt livekit-client ESM ──────────────────────────────────
LIVEKIT_SRC="node_modules/livekit-client/dist/livekit-client.esm.mjs"
if [[ ! -f "${LIVEKIT_SRC}" ]]; then
    echo "error: ${LIVEKIT_SRC} not found after npm install" >&2
    exit 1
fi
cp "${LIVEKIT_SRC}" "${OUT_LIVEKIT}"
echo "Copied $(basename "${OUT_LIVEKIT}")  ($(stat -c%s "${OUT_LIVEKIT}") bytes)"

printf '%s\n' "${VERSION}" > "${VERSION_MARKER}.partial"
mv "${VERSION_MARKER}.partial" "${VERSION_MARKER}"
printf '%s\n' "${LIVEKIT_VERSION}" > "${LIVEKIT_VERSION_MARKER}.partial"
mv "${LIVEKIT_VERSION_MARKER}.partial" "${LIVEKIT_VERSION_MARKER}"

echo
echo "Done. Vendor bundles ready under ${VENDOR_DIR}/."
