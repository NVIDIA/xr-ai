#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: ./play_recording.sh <recordings_dir>
# Plays all .264 chunk files in a participant recording directory in order.
#
# <recordings_dir> is the per-participant folder, e.g.:
#   /tmp/xr_recordings/mcp-agent/web-client

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <recordings_dir>"
    echo "  e.g. $0 /tmp/xr_recordings/mcp-agent/web-client"
    exit 1
fi

DIR="$1"

if [[ ! -d "$DIR" ]]; then
    echo "Error: directory not found: $DIR"
    exit 1
fi

mapfile -t FILES < <(ls "$DIR"/*.264 2>/dev/null | sort -V)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No .264 files found in $DIR"
    exit 1
fi

echo "Found ${#FILES[@]} chunk(s) in $DIR"

TMPLIST=$(mktemp /tmp/chunks_XXXXXX.txt)
TMPMP4=$(mktemp /tmp/recording_XXXXXX.mp4)
trap 'rm -f "$TMPLIST" "$TMPMP4"' EXIT

for f in "${FILES[@]}"; do
    printf "file '%s'\n" "$f"
done > "$TMPLIST"

echo "Muxing ${#FILES[@]} chunk(s)..."
ffmpeg -f concat -safe 0 -i "$TMPLIST" -c:v copy -y "$TMPMP4"

echo "Playing..."
ffplay -autoexit "$TMPMP4"
