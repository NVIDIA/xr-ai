<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Contributing and conventions

Refer to the root
[`CONTRIBUTING.md`](https://github.com/NVIDIA/xr-ai/blob/main/CONTRIBUTING.md)
for setup, testing, pull requests, and DCO sign-off. Refer to
[`AGENTS.md`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md) for the
architecture and change constraints used by humans and coding agents. Refer to
{doc}`Documentation style <documentation-style>` for requirements governing
customer-facing Markdown and reStructuredText. Refer to
{doc}`Python code style <python-code-style>` for public API, exception handling,
and module layout conventions, including `Raises:` exclusions for caller
mistakes. Refer to
[`DEPENDENCIES.md`](https://github.com/NVIDIA/xr-ai/blob/main/DEPENDENCIES.md)
for the package dependency map.

For a Python change:

```bash
uv --config-file uv.toml sync --project <affected-project>
uv --config-file uv.toml run --project tests pytest <affected-test-files>
uv tool run ruff check <changed-python-files>
```

## Change contract

New public names and wider exported contracts need a concrete consumer and
maintainer agreement. Default new helpers to private.

After changing or removing a name, path, flag, port, default, or behavior,
search `docs/source/`, READMEs, configuration and YAML comments, docstrings,
port tables, `skills/`, `.github/workflows/`, and `sonar-project.properties`.
Update affected references in the same PR.

Public Python references are generated from each enrolled module's literal
`__all__`, declarations, annotations, defaults, and docstrings. An API-only
change therefore updates the code, its co-located documentation, and tests; the
strict documentation build rejects unresolved or undocumented exports. Update
a narrative page when concepts, workflows, operations, or architecture change,
and add a migration entry for a breaking change. Refer to
{ref}`migration-release-rollover` for the migration guide's release scope.

Ruff enforces exception-handling rules (`BLE001`, `S110`, `S112`, `B904`) on
all new code. Violations that predate the rules are grandfathered in the
generated `ruff-baseline.toml`, which records the exact count per file and rule
and can only shrink. When a change fixes violations, run
`uv run --script .github/scripts/ruff_baseline.py --write` and commit the
lowered entries. Running the script without `--write` checks that the baseline
matches the code and prints how much of the repository each rule covers; the
pre-commit hook runs that check when Python files change. To keep
a deliberate broad catch in new code, add an inline `# noqa: BLE001` comment with
the reason.

(repository-documentation)=
## Repository documentation

`docs/source/` is the canonical home for user-facing concepts, architecture,
workflows, operations, troubleshooting, and reference material. Do not duplicate
that content in a README. A README is a concise repository entry point: state
what the directory contains, link to the canonical documentation, and keep only
the minimum local information needed before following that link. A top-level
sample README under `agent-samples/` or `model-server-samples/` additionally keeps
the exact commands needed to start that sample. State the sample directory as
the working directory and write every command relative to it, including commands
that invoke a nested project such as `model-servers` from an agent sample. It
also includes a compact configuration entry point: name the sample-owned YAML
or JSON files, describe the common settings they own, and show one representative
edit. Put exhaustive field descriptions, precedence, path resolution,
architecture, behavior, artifact contracts, extension instructions, and eval
methodology in the sample's `docs/source/` guide. Nested READMEs may contain
only artifact-local instructions that are not useful in the published
documentation.

When documentation exists in both locations, consolidate it into
`docs/source/` and replace the README copy with a link. When moving README
content, preserve any still-valid operational detail in the destination page;
do not delete unique guidance merely to shorten the README. Package API details
remain in declarations and co-located docstrings and are rendered into the
generated Python reference rather than repeated in either narrative location.

The user-facing command catalog is generated from top-level sample
`[project.scripts]` entries and literal `argparse` declarations. Keep option
descriptions in `help=` and do not repeat flag tables in narrative pages.

Sample configuration examples live in checked-in YAML and JSON files, with
field-level guidance in adjacent YAML comments. The
generated catalog enrolls files under a top-level sample's `yaml/` tree and
files beside a direct capability subproject, then renders them verbatim.
Narrative docs cover only
workflows, operational decisions, credentials, and process relationships.

## Project metadata

Application-owned projects under `apps/` are outside the repository sample and
contribution tooling described below. After a `pyproject.toml` change elsewhere,
run
`uv run --script .github/scripts/generate_dependency_map.py`. The pre-commit hook
normally regenerates the Python inventory in `DEPENDENCIES.md` automatically,
and CI rejects drift. Do not edit that generated section by hand. Regenerate the
affected project's gitignored `uv.lock` locally. New source files outside
`apps/` require an SPDX header; refer to {doc}`SPDX headers <spdx-headers>`.

(dependency-qualification)=
## Dependency qualification

The root `uv.toml` records the repository's dependency qualification cutoff.
Repository CI passes it explicitly to uv, so fresh resolutions ignore
package-index artifacts uploaded after that timestamp while normal
compatibility ranges stay in package metadata. Standalone nested projects do
not always inherit the root file: uv stops upward config discovery at a nearer
`[tool.uv]` table, and most projects define `[tool.uv.sources]`. Run uv from the
repository root with `--config-file uv.toml` as shown above. To qualify newer
dependencies, advance the cutoff in a dedicated change, resolve every project
with `uv --config-file uv.toml lock --upgrade --project <directory>`, and run
the full test suite. The generated per-project lockfiles are validation
artifacts and remain gitignored. The same change regenerates the committed
locks under `dependency-manifest/`; the pre-commit hook does the Python one when
`uv.toml` is staged. Ordinary dependency or project changes remain deferred to
the next qualification or targeted security refresh, even when a new project
is missing from the manifest: it records qualified dependencies, not each PR's
current resolution. The generator explicitly excludes Speaker STT because its
NeMo Speech sources conflict with the other ASR services; validate its local
lock separately.

For a targeted security refresh, add an `exclude-newer-package` timestamp in
`uv.toml` for each fixed package and add its minimum safe version to the
projects whose locks actually resolve it. This qualifies the fix without
advancing the global cutoff. Resolve each affected project with the root config
explicitly supplied, validate the affected behavior, and regenerate the
dependency manifest with
`uv run --script .github/scripts/generate_dependency_manifest.py`.
Commit the generated manifest changes alongside the dependency declarations.

(dependency-api-compatibility)=
## Dependency API compatibility

This compatibility guidance applies to both repository and consumer code.
Dependency minimums must supply every API the code uses, including optional
SDK APIs. Standard-library calls must work on the minimum supported Python.
Verify API availability from the minimum version's API surface or by running
the affected tests with that version. Separately, validate dependency resolution
and the changed behavior using the project's lock or resolution workflow. A
newer resolved version does not prove the declared floor.

(pull-request-scope)=
## Pull request scope and descriptions

Keep one concern per NVIDIA/xr-ai pull request, ideally under 300 changed lines,
across all languages and docs. For a larger change, justify its size in the PR
description. Leave unrelated pre-existing defects to follow-ups. Confirm the
intended behavior with the user before implementing review requests that expand
scope, and state it in the PR description.

State the problem, resulting behavior, and whether the change is a refactor.
For validation, give reproducible commands, local or CI execution, results, and
relevant hardware or toolchain versions. State what was not run. A native build
or mocked test does not establish physical-device or GPU behavior. Keep these
claims current when scope changes; link CI runs when citing CI results.

Keep review replies brief: state the disposition, correction, and new evidence.
Explain disagreements using the intended contract before changing behavior.

(resource-lifecycle)=
## Resource lifecycle

These lifecycle practices apply to Python, Kotlin, Swift, and browser code in
both XR AI and consumer projects. One owner acquires, uses, and releases each
task, process, track, or native handle. Cleanup after replacement must check
identity before removing shared state. Synchronization covers acquisition,
use, and release whenever they share mutable state.

Make teardown exception-safe so a failed cleanup step does not skip remaining
owned resources. Unpublishing a native track does not by itself establish that
its capture resources are disposed. Await only completion events guaranteed by
the SDK contract, and bound waits so failure remains observable.

In Python, a timeout or cancellation of `asyncio.to_thread()` does not stop an
already running thread. Keep its resource ownership and serialization in force
until the underlying operation finishes; a timed-out await cannot authorize a
concurrent close or reuse. If the thread acquires a resource after its caller
is cancelled, the returned handle can be lost. Have the worker record ownership
or release the resource itself so cleanup does not depend on that return value.

Budget the full shutdown sequence, including drains and joins, within the
owning supervisor's deadline. The repository launcher's `_STOP_TIMEOUT` in
[`xr_ai_launcher/_stack.py`](https://github.com/NVIDIA/xr-ai/blob/main/utils/xr-ai-launcher/xr_ai_launcher/_stack.py)
allows 20 seconds before SIGKILL; sequential cleanup timeouts must fit that total.

(migration-release-rollover)=
## Migration entries and release rollover

{doc}`/reference/migrations` covers changes from the latest published release
to the checked-out version. Use a `Changes since <baseline tag>` heading and
state that `main` is unreleased. Add new entries at the top of that heading's
third-level entries, before `Earlier releases`. Link to the baseline release's
archived guide and {doc}`/reference/migration-history`; users skipping releases
apply earlier guides in release order.

A breaking change requires user or operator action to preserve existing
supported behavior. Consider incompatible changes to APIs, CLI flags and exit
statuses, ports, default-stack contents, configuration keys and defaults, and
wire messages between separately deployed peers. Include required operator
actions such as stopping services with the old checkout before updating or
restarting a reused listener. Compatibility with a still-running older peer is
a real deployment case.

For each breaking change, name who is affected, state the required action, and
include a before-and-after example when useful. Update the entry when another
change in the same development cycle supersedes it, so the instructions lead
directly from the baseline release to the final behavior.

A feature first added since the baseline release needs no migration entry for
its addition, rename, or removal unless it also affects a released contract or
requires action for a still-running older peer. Pre-merge PR revisions never
count as released behavior. Check the baseline tag before adding compatibility
code for an old name; it is needed only for released contracts or a supported
independently deployed peer. Refer to {doc}`documentation-style` for published
URL fragments.

Use this checklist for every release:

1. Prepare the guide on the branch being released. Compare its code with the
   previous release, consolidate superseded instructions, and include only
   changes shipped by this release. State the release version in the opening
   paragraph. Release branches can contain a different set of changes from
   `main`.
2. Preserve the completed guide under
   `docs/source/reference/migration-history/<from-tag>-to-<to-tag>.md`, with a
   title and introduction naming both releases. Copy from the branch being
   released and retain the upgrade instructions. Scope explicit labels to the
   archive and pin historical cross-references to the release's tagged
   documentation so current topics cannot change their meaning. Add the guide
   to the toctree in `docs/source/reference/migration-history.md`.
3. Publish the release tag and its versioned documentation. Verify that the
   archived guide is published, its historical links resolve, and
   `https://nvidia.github.io/xr-ai/<tag>/reference/migrations.html` contains the
   completed guide. Use a concrete tag in historical links; `latest` and `main`
   change over time.
4. After verifying publication, bring the completed archive and its index entry
   into `main`, then reconcile the current entries with the tag. Use
   `git merge-base <tag> main` to establish shared history and
   `git diff <tag> main -- docs/source/reference/migrations.md` to compare
   guides, then verify the corresponding code changes, including equivalent
   cherry-picks. Retire an entry only when the published tag covers its change
   and the archive retains its upgrade instructions. Keep main-only entries.
   Carry a release-branch entry into the current guide only when its code is
   also present on `main`. Update the baseline and archive link to the new
   release, state that `main` is unreleased, and mark the current guide empty
   only if no entries remain. Keep completed archives unchanged.
5. Preserve published URL fragments with compact forwarding links to the
   appropriate archived guide or canonical topic. Keep ongoing explanations in
   canonical documentation and update their incoming references before retiring
   the entry.

When porting changes to a release branch, apply dependent commits in ancestry
order. Compare each resulting patch with its source commit and inspect conflict
resolutions for unrelated edits or release-specific behavior lost in the port.
