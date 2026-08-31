---
name: gh-develop-xr-ai
description: Develop and update NVIDIA XR-AI pull requests with small, explicit scope, complete tests and documentation, accurate descriptions, tracked follow-up chains, an isolated self-review, and reasoned handling of reviewer feedback. Use when implementing a change for an XR-AI PR, opening or updating that PR, planning a sequence of XR-AI PRs, or addressing review comments on authored work.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Develop an XR-AI pull request

Deliver one independently useful change that is easy to understand, validate,
and review. Keep the current PR complete without pulling future work into it.

## 1. Establish the outcome

Read the repository `AGENTS.md`, the nearest README, canonical documentation,
and affected tests before editing. Write a one-sentence outcome and list:

- behavior that must change;
- tests and documentation required to make that behavior complete;
- deliberate exclusions; and
- independently useful work that belongs later.

Remove unrelated cleanup from the plan. Do not hide required current behavior
behind a follow-up.

If the request intentionally begins a series, create one GitHub issue for each
remaining independently reviewable outcome. Use the authenticated GitHub
account to create and assign each issue to that same account. Give every issue
a narrow outcome, affected area, and validation target. Link the issues from
the current PR and explain where this PR sits in the sequence. Do not create a
tracking issue when the current PR is self-contained.

## 2. Implement a self-contained change

Change only what the stated outcome needs. Include the implementation, focused
tests, dependency metadata, migration notes, and user-facing documentation that
must land atomically. Follow repository generation and validation rules for
the affected files.

Keep opportunistic refactors, formatting churn, broad abstractions, and
unrelated documentation corrections out of the diff. If a discovered problem
is material but independent, raise it separately instead of expanding the
current PR. Create an issue only after it becomes accepted, planned work in a
follow-up series.

Validate during implementation with the narrowest relevant checks, then run
the broader repository checks warranted by the risk. Record exact commands and
honestly state skipped or unavailable checks.

## 3. Perform an isolated self-review

Before requesting human review, stop implementation and review the complete
merge-base diff at least once from a fresh reviewer perspective. Use a separate
context or reviewer when available; otherwise begin with only the outcome, PR
description, and diff rather than the coding notes.

During the first pass, inspect without editing and record findings:

1. Compare every changed line with the stated problem and remove unnecessary
   churn.
2. Trace important success, failure, compatibility, lifecycle, and cleanup
   paths against the surrounding implementation.
3. Confirm tests would fail without the intended change and cover meaningful
   failure behavior.
4. Confirm documentation describes the current behavior, paths, commands,
   defaults, and deliberate constraints.
5. Check that the PR description accounts for every material part of the diff
   and does not promise future work as current behavior.

Fix valid findings, rerun affected validation, and repeat the diff check after
substantial changes.

## 4. Write the PR description

Keep the repository template headings and make each section concrete:

- **Problem:** State the observable problem and why this PR is needed.
- **Solution:** Explain how this diff solves that problem, including important
  implementation choices rather than a file list.
- **Scope and follow-ups:** State deliberate exclusions, constraints, and
  tradeoffs. For a series, explain the current step and link the assigned
  follow-up issues. Otherwise write that no follow-up is planned.
- **Validation:** List exact checks run and relevant checks not run.

Update the description whenever the implementation or scope changes. Give
reviewers enough context to recognize deliberate choices without requiring
them to reconstruct the development history.

## 5. Handle reviewer feedback

Refresh the current head, comments, reviews, and checks before responding. For
each request:

1. Restate the underlying concern and reproduce or verify it against the code.
2. Classify it as a current correctness gap, a small high-value improvement, an
   alternative implementation preference, or unrelated/follow-up work.
3. Apply the smallest correct fix that preserves the PR's stated outcome.
4. Rerun focused validation and inspect the resulting diff for new churn.
5. Reply with a concise disposition: fixed and how, declined and why, or
   deferred to accepted follow-up work with a linked issue.

Do not implement a suggestion only because a reviewer prescribed it. Accept
simple, high-value fixes that strengthen the current outcome. Do not turn the
PR into a full redesign or unrelated cleanup; if feedback proves the stated
approach fundamentally incorrect, stop and ask whether to replace, narrow, or
split the PR rather than silently changing its purpose.

Before requesting re-review, update the PR description and repeat the isolated
self-review for the reviewer-driven diff.
