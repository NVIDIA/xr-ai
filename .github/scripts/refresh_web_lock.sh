#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Regenerate dependency-manifest/web-xr-build/package-lock.json from the current
# client-samples/web-xr-build/package.json and the pinned CloudXR tarball. Lock
# output varies across npm releases, so the lock is always written by NPM_VERSION.

set -Eeuo pipefail

NPM_VERSION="10.8.2"

cd "$(dirname "$0")/../.."
SOURCE="client-samples/web-xr-build"
TARGET="dependency-manifest/web-xr-build"

version="$(tr -d '[:space:]' < "${SOURCE}/.sdk-version")"
file="nvidia-cloudxr-${version}.tgz"
if [[ ! -f "${TARGET}/sdk.tgz" ]] || ! tar -xOzf "${TARGET}/sdk.tgz" package/package.json | grep -q "\"version\": \"${version}\""; then
    curl -fsSL --output "${TARGET}/sdk.tgz" \
        "https://api.ngc.nvidia.com/v2/resources/org/nvidia/cloudxr-js/${version}/files?redirect=true&path=${file}"
fi

cp "${SOURCE}/package.json" "${TARGET}/package.json"
cd "${TARGET}"
npx --yes "npm@${NPM_VERSION}" install --package-lock-only --ignore-scripts --no-audit --no-fund --legacy-peer-deps
