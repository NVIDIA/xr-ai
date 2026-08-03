#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remove one PR's documentation preview from the ``gh-pages`` branch.
#
#   remove_docs_preview.sh <pr-number>
#
# Idempotent: a missing branch or a missing directory is success, so it is safe
# to call speculatively. Callers must serialize with publish_docs.sh — both
# push the same branch.
set -euo pipefail

pr="${1:?usage: remove_docs_preview.sh <pr-number>}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

remote="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

if ! git ls-remote --exit-code --heads "$remote" gh-pages >/dev/null 2>&1; then
  echo "No gh-pages branch; nothing to remove."
  exit 0
fi

work="$(mktemp -d)"
git clone --quiet --depth 1 --branch gh-pages "$remote" "$work"

if [[ ! -d "$work/preview/pr-${pr}" ]]; then
  echo "No preview for PR #${pr}."
  exit 0
fi

cd "$work"
rm -rf "preview/pr-${pr}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add --all
git commit --quiet --message "docs: drop preview for PR #${pr}"

for attempt in 1 2 3; do
  if git push --quiet origin gh-pages; then
    echo "Removed preview for PR #${pr}."
    exit 0
  fi
  echo "Push rejected (attempt ${attempt}); rebasing onto the current branch."
  git fetch --quiet --depth 1 origin gh-pages
  git rebase --quiet FETCH_HEAD
done

echo "Could not remove the preview for PR #${pr} after 3 attempts." >&2
exit 1
