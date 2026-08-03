#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Publish a built documentation site to the ``gh-pages`` branch.
#
#   publish_docs.sh <site-dir>          publish to the branch root
#   PR_NUMBER=42 publish_docs.sh <dir>  publish to preview/pr-42/
#
# Publishing the site replaces everything at the root except ``preview/``, so a
# deleted release tag does not leave a stale directory behind while open PR
# previews survive. Publishing a preview touches only that PR's directory.
#
# Callers must serialize: every invocation pushes the same branch. The retry
# loop only covers a lost race, not concurrent use as a general strategy.
set -euo pipefail

src="${1:?usage: publish_docs.sh <site-dir>}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
pr="${PR_NUMBER:-}"

remote="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
work="$(mktemp -d)"

if git ls-remote --exit-code --heads "$remote" gh-pages >/dev/null 2>&1; then
  git clone --quiet --depth 1 --branch gh-pages "$remote" "$work"
else
  # First publish: start an empty branch rather than branching off the default
  # one, so the site never carries the source tree's history.
  git init --quiet --initial-branch gh-pages "$work"
  git -C "$work" remote add origin "$remote"
fi

if [[ -n "$pr" ]]; then
  dest="$work/preview/pr-${pr}"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -R "$src/." "$dest/"
  message="docs: preview for PR #${pr}"
else
  find "$work" -mindepth 1 -maxdepth 1 ! -name .git ! -name preview -exec rm -rf {} +
  cp -R "$src/." "$work/"
  message="docs: publish ${GITHUB_SHA:0:8}"
fi

# Sphinx writes .nojekyll per build directory; the branch root needs its own or
# Jekyll drops every _static/ path.
touch "$work/.nojekyll"

cd "$work"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add --all

if git diff --cached --quiet; then
  echo "Documentation is unchanged; nothing to publish."
  exit 0
fi

git commit --quiet --message "$message"

for attempt in 1 2 3; do
  if git push --quiet origin gh-pages; then
    exit 0
  fi
  echo "Push rejected (attempt ${attempt}); rebasing onto the current branch."
  git fetch --quiet --depth 1 origin gh-pages
  git rebase --quiet FETCH_HEAD
done

echo "Could not publish to gh-pages after 3 attempts." >&2
exit 1
