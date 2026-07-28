<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# PR & review guidelines — NAT capability alignment

These are the standards applied to the `xr-ai-nat` capability-alignment PRs (the
`nat-align/*` series) as `xr-ai` converges toward its modular target. They are
distilled from the review threads on those PRs and are meant to be applied by
both authors and reviewers so each PR lands in one or two rounds instead of many.

Most of these are simply good practice for any change that folds/renames a
public surface, touches an agent-facing tool schema, or depends on another
in-flight PR — reach for them beyond the `nat-align/*` series too.

## Principles

1. **Never silently reverse a landed PR.** Behavior preservation is the top bar.
2. **Prefer `main`'s stricter contracts over the fork's leaner ones** — keep
   validators, field constraints, and descriptions rather than copying every
   fork simplification.
3. **The agent-facing surface is a contract, not cosmetic** — the NAT-generated
   tool input schema *and* the function/field descriptions are what the model
   consumes; treat changes to them as interface changes.
4. **Public surfaces stay backward-compatible** via *complete, tested* deprecated
   aliases.
5. **Stay integrated with `main`** — rebase, use canonical imports, reconcile docs.
6. **Keep PRs concise and single-purpose** — no dead or unconnected surface.
7. **Verify with evidence, and re-verify after integration** — green checks that
   predate a rebase do not count.

## Checklist

### A. Do-not-reverse audit
- Audit the change against each landed PR **by number**, and cite the guarding
  test (e.g. the video-mcp conditional tool sets keyed on
  `VideoHealthResult.recording_enabled`; the transcript-store `_check`
  path-escape guard and its symlink test).
- Keep `main`'s field constraints and validators (`gt=0`, `min_length=1`, …) and
  its agent-facing descriptions over the fork's leaner models.

### B. Agent-facing NAT tool schema
- Register a **thin strict native wrapper** whose parameter is a **required**
  typed request. Do **not** register a client method whose Python
  back-compat parameter (`request: Req | None = None`) leaks an optional/nullable
  wrapper into the generated schema. A no-input operation must generate a
  **strict empty object** (`{}`), never `{"request": {"anyOf": [Req, null]}}`.
- Add a regression test that inspects the **generated** schema
  (`input_schema.model_json_schema()`) and asserts the shape (no nullable
  `request` wrapper; `properties == {}` for no-input ops).
- Keep the detailed function / config / field **descriptions** — call-ordering
  constraints ("discovery before recorded queries"), absolute-vs-relative
  semantics, "never a live-camera frame", endpoint/timeout meaning. Do not
  shorten them away.

### C. Backward-compatible public surface
- When folding or renaming, keep a **complete** deprecated forwarding-alias
  module: re-export the unchanged names **and** alias every renamed-but-
  same-contract class (verify the field contracts are actually identical), keep
  the package-level exports, and emit a `DeprecationWarning` on import.
- Keep legacy / no-argument call forms **on the client** (build the typed request
  internally, e.g. `list_recorded_participants(request=None)` →
  `ListRecordedParticipantsRequest()`), separate from the strict agent wrapper.
- Genuinely-removed concepts with no equivalent (error-as-data patterns such as
  `TextMemoryError`) may be dropped — but document them as removals and confirm
  no remaining consumers.
- Add regression tests for the **legacy names** and the **compat call forms**.

### D. Integration with `main`
- Rebase onto current `main` whenever a dependency PR lands. Import the
  **canonical** path (`xr_ai_nat.functions._service.rpc`, `xr_ai_nat.mcp`), not a
  deprecated alias. Reconcile any changelog / doc overlap with the merged PRs.
- **Re-run** the focused suite and CI *after* integrating — green checks that
  predate the integration are stale.

### E. Scope & focus
- **Defer** any group/capability with no consumer or producer on `main`;
  registering it is dead, unconnected surface. Land it later **with its real
  consumer and an end-to-end test**.
- Route structured capability results the model needs (health / readiness,
  including flags like `recording_enabled`) through the **native toolbox or
  injected capability context — not through MCP**.

### F. Tests
- Add **real-signature** happy-path tests (a real client against an in-memory
  server, not a permissive stub). Fix any stub whose looser signature would mask
  a regression — that is exactly how the `list_recorded_participants` `TypeError`
  slipped past the original tests.

### G. Docs & hygiene
- The changelog must match the **actual** code — correct import paths, and the
  correct merged/open status of related PRs (no stale "separate open PR" wording
  after the dependency merges).
- Keep the **PR description in sync** with the final branch: what is included,
  what is deferred, what was rebased. A stale description reads as unfocused.
- Do not import a module via both `import X` and `from X import y`.

## References

- Convergence target: `nvddr/xr-ai-ddr-fork@agent/modular-render-subagents`.
- Worked examples: the review threads on the `nat-align/*` video-memory and
  text-memory PRs, where each of the above was raised and resolved.
