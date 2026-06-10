<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo — Working Conventions

How to run and iterate on this sample's prompt eval efficiently. For what the
harness covers and how to write a case, see `eval/README.md`.

## Running the eval (use the watcher)

The user starts `eval/eval_watch.sh` **once**; your loop is then edit
`worker/prompts/system.txt` → wait for the debounce → read the log.

- Never run `eval.py` or `eval_watch.sh` yourself. The user approved the
  watcher once; re-running it re-asks for permission and defeats the
  already-running watcher (it also hangs under sandboxing). If a watcher is
  running, just edit the prompt.
- The trigger is a **sha1 of `system.txt`'s content bytes, not its mtime**
  (~10s debounce); results land in the gitignored `eval/.eval_loop.log`. The
  newest run block has per-case `✓`/`✗` and an `N/M passed` line, plus a
  `FILTER:` line when `eval/.only` is active (absent on a full run).
- A bare `touch` (mtime-only) will **not** fire a run — make a real content
  edit. To rerun without changing model-visible text, toggle a trailing
  whitespace line in `system.txt` (the eval `.strip()`s the prompt), then
  remove it. Editing `eval.py` does not trigger a run (only `system.txt` is
  watched).
- Scope which cases run via `eval/.only` (newline- or comma-separated case
  names; `#` comments OK).

## Iteration strategy

The eval is **deterministic per prompt bytes**: identical `system.txt` yields
identical per-case results. Use that.

- Iterate on a focused `eval/.only` of the actual reds plus any cases known to
  flip-flop. A case green on the subset stays green in the full suite for the
  same bytes. Run the full case suite **only once, at the end, to confirm**.
- When a confirming full run surfaces **new collateral reds** (cases green
  before that a prompt-length/byte change knocked off), add those names to
  `eval/.only` and resume fast subset iteration on the expanded set — don't
  re-run the full suite to re-discover them. Let `.only` grow to cover every
  flip-prone case you've seen so the subset stays representative.
- **One change per run.** Save, wait for the fresh `passed` line, read it, then
  edit again. Editing faster than the watcher runs just scores bytes you've
  already abandoned — what looks like "flakiness" is almost always this.
- **Don't go silent.** Poll the log with bounded waits rather than sitting in a
  long thinking turn. Act, observe, act.

## Priorities (in order)

1. **Protect cases that are already green** — especially the colour
   say==built / one-mutation wins. Never trade a held win for a new case.
2. **Fix regressions** (green before your change, red after) before anything
   else.
3. **Pre-existing baseline reds** (red before any of your edits) are findings,
   not your bug. Record them; do not distort the prompt to chase them.
4. **Harness bug vs model failure.** If the model behaved correctly and the
   check is wrong (e.g. a valid compound colour like "blue‑green", or a
   non-ASCII hyphen), fix the harness — not the prompt.

## Prompt hygiene

- **Shorter is safer.** Prompt bloat reshuffles byte-sensitive spatial cases;
  prefer the shortest wording that holds the behaviour.
- **Positive phrasing.** "Do X" beats "never do Y" — negation-priming makes
  this model perform the forbidden action.
- **No internal slang.** Bug nicknames mean nothing to the model — state the
  plain instruction.
