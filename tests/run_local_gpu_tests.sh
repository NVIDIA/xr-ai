#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the gpu-marked tests against the local box. These tests need a real
# GPU / Docker / NVENC and are filtered out of GitHub CI.

set -euo pipefail
cd "$(dirname "$0")"
uv --config-file ../uv.toml sync
uv --config-file ../uv.toml run pytest -v --tb=short --color=yes -m gpu "$@"
