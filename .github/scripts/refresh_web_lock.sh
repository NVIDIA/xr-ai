#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Regenerate dependency-manifest/web-xr-build/package-lock.json at the uv.toml
# cutoff. Lock output varies across npm releases, so NPM_VERSION always writes it.

set -Eeuo pipefail

NPM_VERSION="10.8.2"

cd "$(dirname "$0")/../.."
SOURCE="client-samples/web-xr-build"
TARGET="dependency-manifest/web-xr-build"

cutoff="$(sed -nE 's/^exclude-newer = "([^"]+)"$/\1/p' uv.toml)"
[[ -n "${cutoff}" ]] || { echo "uv.toml has no exclude-newer cutoff" >&2; exit 1; }

cp "${SOURCE}/package.json" "${TARGET}/package.json"

# Resolve in an empty directory so nothing carries over from the committed lock.
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cp "${TARGET}/package.json" "${work}/"
(
    cd "${work}"
    npx --yes "npm@${NPM_VERSION}" install --package-lock-only --ignore-scripts --no-audit --no-fund \
        --before="${cutoff}"
)
cp "${work}/package-lock.json" "${TARGET}/package-lock.json"
