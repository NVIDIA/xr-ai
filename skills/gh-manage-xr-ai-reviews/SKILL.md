---
name: gh-manage-xr-ai-reviews
description: Manage a user-provided inbox of NVIDIA XR-AI pull request reviews and provide convenient overall status. Use when the user sends multiple PRs to review, asks what reviews remain, requests a review-status summary, asks to check for new feedback or commits, resumes several in-flight reviews, or authorizes a batch of review comments. Track each PR independently from intake through re-review, refresh GitHub state, use bounded subagents for analysis, and apply the comment-only XR-AI review contract without ever approving automatically.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Manage the XR-AI review inbox

Give the user one coherent view of all PR reviews they have sent. Preserve the
state and next action for each PR while keeping each actual review isolated.

## Apply the per-PR review contract

Read and follow `../gh-review-xr-ai/SKILL.md` completely before analyzing or
posting feedback on any PR. Its scope, independent-verification,
classification, and comment-only rules apply separately to every inbox item.

Never approve, request changes, post inline feedback, or alter an existing
review. Draft by default. Treat posting authorization as PR-specific unless the
user explicitly names a batch. Authorization covers only the exact draft and
head SHA shown to the user; invalidate it when the head changes or the draft
changes materially.

## Accept PRs into the inbox

When the user supplies PR numbers or URLs, add only those PRs. Do not silently
expand the inbox to every open repository PR. Resolve ambiguous numbers to the
current XR-AI repository and report any PR that cannot be found.

On intake, fetch:

- number, URL, author, title, draft or ready state, labels, base, and head SHA;
- changed-file count and primary component;
- requested reviewers and existing review states;
- unresolved review threads and author replies;
- CI/check status;
- dependencies or conflicts stated by the PR.

Read the description and code only when beginning review analysis. Intake
should stay fast enough to provide the user an immediate queue summary.

## Track a useful lifecycle

Keep one isolated record per PR with its reviewed head SHA, last refresh, draft
findings, posted review URL, and next action. Use one of these states:

- `queued`: received but not yet reviewed;
- `reviewing`: analysis is active;
- `draft-ready`: review is complete but not posted;
- `waiting-author`: feedback was posted and no relevant update has arrived;
- `needs-rereview`: new commits or replies require another pass;
- `blocked`: review cannot proceed for a stated external reason;
- `done`: current head was reviewed and no action remains.

Track CI separately as `pending`, `passing`, `failing`, `skipped`, or
`unknown`. CI status does not replace review state: a `draft-ready` review can
have pending CI, and a `waiting-author` review can have failing CI.

State transitions must be evidence-based:

- A changed head after review becomes `needs-rereview`.
- An author reply alone becomes `needs-rereview` only if it answers or disputes
  a finding; otherwise record it without reopening the review.
- A posted review becomes `waiting-author` while its findings remain open.
- Resolved feedback on an unchanged head may become `done` after verification.
- A green check does not make an unreviewed PR `done`.

Never call a PR approved by this workflow. Report other reviewers' approval as
external GitHub state, not as this review's conclusion.

Carry the inbox forward in each overall status response so it survives normal
conversation summarization. If prior inbox state is unavailable, reconstruct
known items from the latest visible status and current GitHub data. If the PR
list itself cannot be recovered, ask the user for it; never guess by importing
all open PRs. Keep unposted draft bodies in working context; a status label is
not a substitute for the draft. If a `draft-ready` body cannot be recovered,
change the item to `reviewing`, regenerate the review against the current head,
show the replacement draft, and obtain fresh posting authorization. Do not
create repository tracking files for inbox persistence.

## Refresh before answering status questions

When the user asks “what remains,” “more reviews,” “check feedback,” “status,”
or similar, refresh all active inbox PRs before answering. Compare current
state with the stored head, reviews, threads, comments, and checks. Highlight
only meaningful deltas:

- new commits;
- new or dismissed reviews;
- actionable comments or replies;
- thread resolution changes;
- CI transitions;
- draft/ready or merge-state changes.

Avoid re-reading complete diffs for unchanged PRs. Re-review changed PRs using
the new commits plus the cumulative merge-base-to-head three-dot diff and
affected neighboring code.

## Provide an action-oriented status view

Lead status responses with a compact rollup, for example:

`2 need review · 1 draft ready · 2 waiting on authors | CI: 1 pending, 1 failing`

Keep review-state counts disjoint. Report CI counts separately because they may
overlap any review state.

Then show the inbox:

| PR | Author | Existing review | CI | Our state | Last change | Next action |
|---|---|---|---|---|---|---|

Keep “Existing review” factual: summarize active human review states and
unresolved threads without treating bot output as human approval. Make “Our
state” and “Next action” explicit so the user can decide what to do without
asking a second question.

Sort the table by actionability:

1. `needs-rereview`;
2. `draft-ready`;
3. `queued` or `reviewing`;
4. `blocked`;
5. `waiting-author`;
6. `done`.

Within each state, surface failing CI before pending CI and pending CI before
passing CI.

Omit completed items from routine summaries when the list is long, but include
their count and show them when requested.

## Make common review requests convenient

Interpret common requests consistently:

- **“Review 404, 403.”** Add both, refresh them, review independently, and
  return per-PR drafts plus the overall inbox status. Do not post.
- **“Post these.”** Post only the reviews unambiguously referenced by the
  immediately preceding drafts, after refreshing each PR.
- **“More reviews” or “look at additional comments.”** Refresh active items,
  prioritize new author activity and changed heads, and report what changed.
- **“What remains?”** Refresh the inbox and return the rollup, status table,
  and next actions.
- **“Re-review 387.”** Reuse the stored reviewed head and prior findings, then
  inspect new changes and current cumulative behavior before replacing the
  unposted draft. If the acting identity already reviewed the current head,
  report that review unless there are genuinely new findings. Show any
  supplemental draft and require explicit supplemental-post authorization;
  never edit or replace the existing review.
- **“Ignore PRs matching a filter.”** Apply the filter to future summaries and
  retain excluded items only if the user may want them restored later.

After completing review work on multiple PRs, always provide a short overall
status even if the user did not explicitly request one.

## Coordinate inbox parallelism

Apply the per-PR skill's independent-verification workflow separately to each
inbox item. The inbox-specific rules are:

- Keep one coordinator responsible for the inbox, final verification, user
  communication, and every GitHub write.
- Run independent PRs in parallel when capacity permits; use waves when the
  inbox is larger than available capacity.
- Record results under the correct PR number immediately. Never carry a
  finding to another PR without reproducing it against that PR's own base and
  head.

## Handle authorized batches

Apply the per-PR skill's complete refresh, reauthorization, duplicate-review,
and posting checks to every authorized inbox item. After a successful post,
store the posted URL and reviewed head, then update that item's state.

If a batch partially fails, continue only with other explicitly authorized,
independent PRs. Report exactly what posted, what did not, and the resulting
overall status.

Keep unrelated issues out of the active PR review. Suggest a self-contained
follow-up PR and optionally an issue; file the issue only with explicit user
authorization.
