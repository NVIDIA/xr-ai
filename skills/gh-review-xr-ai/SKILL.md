---
name: gh-review-xr-ai
description: Review NVIDIA XR-AI pull requests with strict change-scope discipline. Use when inspecting, re-reviewing, drafting feedback for, or posting a review on an XR-AI PR. Independently verify the PR description, diff, surrounding code, tests, scope, and existing reviews; report only change-caused blockers and nits in one top-level comment; route pre-existing or unrelated problems to self-contained follow-up work; and never approve automatically.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Review an XR-AI pull request

Produce an independent, evidence-backed review of the proposed change. Keep
merge feedback tightly bounded to regressions, risks, and unnecessary changes
introduced by the PR.

## Non-negotiable review contract

- Never approve a PR, submit an `APPROVE` event, or describe the review as an
  approval.
- Never submit `REQUEST_CHANGES`. Post only a normal top-level review comment.
- Never post inline review comments. Include file and line references in the
  single top-level comment instead.
- Never dismiss, alter, or replace another reviewer's approval or review.
- Draft by default. Post only when the user explicitly asks to post or submit
  the review.
- Immediately before posting, refresh the PR and its reviews. Update the draft
  to avoid stale or duplicate feedback, then post one consolidated comment.

## Review workflow

### 1. Establish the review boundary

Read the repository's `AGENTS.md` and applicable nested instructions. Record
the PR's base and head revisions, title, description, labels, linked issues,
commits, changed files, and checks.

Treat the base branch as the control. A review finding belongs in the merge
review only when the PR introduces it, worsens it, or makes it newly reachable.
Do not block the PR on a defect that is unchanged from the base.

### 2. Read existing review activity

Before evaluating the change, inspect all available review evidence:

- top-level PR conversation;
- submitted reviews and their states;
- inline review threads, including resolved or outdated threads when relevant;
- author replies and later commits that may address earlier feedback;
- automated findings, distinguishing tool output from human judgment.

Use other reviews as leads, not conclusions. Independently reproduce each
relevant claim against the current head and actual code. Do not repeat an
existing point unless it remains unresolved and the consolidated review would
otherwise omit a merge-relevant issue. If repeating it, add evidence or clarify
why it is still applicable.

### 3. Verify the description and scope

Compare every material claim in the title and description with the diff and
tests. Check that the description accurately states behavior changes,
motivation, validation, compatibility effects, deployment or GPU implications,
and known limitations when those topics apply.

Audit every changed file for necessity. Flag unrelated refactors, formatting,
renames, generated-file churn, public API expansion, dependency changes, or
behavior changes that are not required for the stated goal. Prefer asking the
author to revert or split unnecessary changes rather than reviewing them as if
they were part of the requested fix.

Expect a PR to be small and coherent. Accept a large scope only when both are
true:

1. The PR carries the repository's explicit large-PR label or tag.
2. Its description convincingly explains why the work cannot be split, maps
   the major change groups to the goal, and documents validation and risk.

Otherwise, treat unjustified size or mixed purpose as a blocker and recommend
independently reviewable PRs.

### 4. Inspect the actual implementation

Read the complete diff, then inspect enough unchanged neighboring code and the
base implementation to understand control flow, ownership, configuration, and
existing patterns. Search for callers, sibling implementations, tests,
documentation, launch paths, and cleanup paths affected by the change.

Evaluate at least:

- correctness and regressions relative to base;
- consistency with established XR-AI structure and ownership boundaries;
- lifecycle, cleanup, port, process, GPU, and model-loading implications when
  applicable;
- backward compatibility and configuration migration;
- failure handling, concurrency, and resource cleanup;
- whether tests exercise the changed behavior and meaningful failure paths;
- whether the implementation adds broader machinery or public API than the
  stated change requires.

Do not propose a new or broader public API unless the user explicitly requests
one. Prefer private implementation details and existing APIs.

### 5. Classify findings by relationship to the change

Use these categories:

- **Blocker:** A PR-introduced correctness bug, regression, security or data
  risk, broken compatibility, misleading material description, missing
  essential validation, unjustified scope, or architectural violation that
  should be fixed before merge.
- **Nit:** A small, actionable, non-blocking issue introduced by the PR. Keep
  nits sparse; do not use them for personal style preferences already allowed
  by repository patterns.
- **Follow-up (not blocking):** A valid pre-existing or unrelated issue that
  should not expand this PR. Describe a self-contained follow-up PR with a
  narrow outcome, affected area, and validation target. If deferral needs
  durable tracking, suggest filing an issue; file it only with explicit user
  authorization.

Do not disguise out-of-scope work as a blocker or nit. Conversely, scope creep
inside the PR is itself change-caused: ask for it to be removed or split.

For each blocker or nit, include:

1. A precise file and line or smallest useful code location.
2. The observable failure or risk.
3. Evidence that the PR causes it relative to base.
4. The smallest viable correction, without prescribing unnecessary redesign.

### 6. Draft one top-level review comment

Use this structure, omitting empty sections:

```markdown
## Blockers

- `path/to/file.py:123` — **Short finding title.** Explain the change-caused
  failure, its impact, the base comparison, and the smallest fix.

## Nits

- `path/to/file.py:145` — Explain the small change-scoped improvement.

## Follow-ups (not blocking)

- Propose a self-contained follow-up PR or an issue with a narrow outcome and
  validation target. State clearly that it does not block this PR.
```

If there are no change-scoped findings, write: `No change-scoped blockers or
nits found.` Do not use approval language such as “Approved,” “LGTM,” or “good
to merge.”

Keep the comment concise. Do not add a generic summary that restates the PR.

### 7. Refresh and post safely

When the user has authorized posting:

1. Re-fetch the current head revision, description, labels, checks, comments,
   reviews, and thread states.
2. If the head changed, re-review the affected diff before posting.
3. Remove findings already fixed or fully covered by another active review;
   preserve independently verified unresolved blockers.
4. Submit exactly one top-level comment-only review. Do not approve, request
   changes, add inline comments, resolve threads, or modify existing reviews.
5. Report the posted review URL to the user.

If posting cannot be guaranteed to use a comment-only event, stop and ask the
user rather than risk submitting an approval.
