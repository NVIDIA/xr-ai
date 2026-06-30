#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# commit-msg hook: verify a Signed-off-by trailer is present.
# pre-commit passes the commit message file path as $1.

set -euo pipefail

if [ -n "${1:-}" ]; then
    msg_file="$1"
else
    git_dir=$(git rev-parse --git-dir 2>/dev/null) || git_dir=".git"
    msg_file="${git_dir}/COMMIT_EDITMSG"
fi

if grep -qP "^Signed-off-by:\s+\S+.*<[^@\s]+@[^@\s]+>" "$msg_file"; then
    exit 0
fi

echo ""
echo "ERROR: Commit is missing a DCO Signed-off-by trailer."
echo ""
echo "  Fix: git commit --amend -s"
echo ""
echo "  Or add to your git config so every commit signs off automatically:"
echo "    git config --global alias.c 'commit -s'"
echo ""
echo "  See https://developercertificate.org/ and AGENTS.md § DCO sign-off."
echo ""
exit 1
