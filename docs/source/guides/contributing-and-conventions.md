<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Contributing and conventions

**TL;DR:** Use this reference when contributing to XR AI so your changes fit
the repository and remain understandable and maintainable for other
contributors across releases.

Refer to the root
[`CONTRIBUTING.md`](https://github.com/NVIDIA/xr-ai/blob/main/CONTRIBUTING.md)
for setup, testing, pull requests, and DCO sign-off. Refer to
[`AGENTS.md`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md) for the
architecture and change constraints used by humans and coding agents. Refer to
{doc}`Documentation style <documentation-style>` for requirements governing
customer-facing Markdown and reStructuredText. Refer to
{doc}`Python code style <python-code-style>` for the public API, exception handling,
and module layout conventions for Python code. Refer to
[`DEPENDENCIES.md`](https://github.com/NVIDIA/xr-ai/blob/main/DEPENDENCIES.md)
for the package dependency map.

For a Python change:

```bash
uv --config-file uv.toml sync --project <affected-project>
uv --config-file uv.toml run --project tests pytest <affected-test-files>
uv tool run ruff check <changed-python-files>
```

Public Python references are generated from each enrolled module's literal
`__all__`, declarations, annotations, defaults, and docstrings. An API-only
change therefore updates the code, its co-located documentation, and tests; the
strict documentation build rejects unresolved or undocumented exports. Update
a narrative page when concepts, workflows, operations, or architecture change,
and add a migration entry for a breaking change to released behavior. Refer to
{ref}`migration-release-rollover` for the migration guide's release scope.

The repository-wide README and documentation ownership rules live in the
[`Comments and documentation`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md#comments-and-documentation)
section of `AGENTS.md`. Follow that contract instead of introducing another
placement rule in a package or sample.

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

The user-facing command catalog is generated from top-level sample
`[project.scripts]` entries and literal `argparse` declarations. Keep option
descriptions in `help=` and do not repeat flag tables in narrative pages.

Sample configuration examples live in checked-in YAML and JSON files, with
field-level guidance in adjacent YAML comments. The
generated catalog enrolls files under a top-level sample's `yaml/` tree and
files beside a direct capability subproject, then renders them verbatim.
Narrative docs cover only
workflows, operational decisions, credentials, and process relationships.

Application-owned projects under `apps/` are outside the repository sample and
contribution tooling described below. After a `pyproject.toml` change elsewhere,
run
`uv run --script .github/scripts/generate_dependency_map.py`. The pre-commit hook
normally regenerates the Python inventory in `DEPENDENCIES.md` automatically,
and CI rejects drift. Do not edit that generated section by hand. Regenerate the
affected project's gitignored `uv.lock` locally. New source files outside
`apps/` require an SPDX header; refer to {doc}`SPDX headers <spdx-headers>`.

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
the next qualification or targeted security refresh. The generator
explicitly excludes Speaker STT because its NeMo Speech sources conflict with
the other ASR services; validate its local lock separately.

For a targeted security refresh, add an `exclude-newer-package` timestamp in
`uv.toml` for each fixed package and add its minimum safe version to the
affected service dependencies. This qualifies the fix without advancing the
global cutoff. Resolve each affected project with the root config explicitly
supplied, validate the affected behavior, and regenerate the dependency
manifest with `uv run --script .github/scripts/generate_dependency_manifest.py`.
Commit the generated manifest changes alongside the dependency declarations.

(migration-release-rollover)=
## Migration entries and release rollover

{doc}`/reference/migrations` covers changes from the latest published release
to the checked-out version. Use a `Changes since <baseline tag>` heading and
state that `main` is unreleased. Add entries as third-level headings under
that heading, before `Earlier releases`. Link to the baseline release's
archived guide and {doc}`/reference/migration-history`; users skipping releases
apply earlier guides in release order.

Before adding an entry, check the baseline tag's code and configuration with
`git show <baseline-tag>:<path>`. Changes affecting only interfaces introduced
after the baseline belong in their canonical documentation.

For each breaking change to released behavior, name who is affected, state the
required action, and include a before-and-after example when useful. Update the
entry when another change in the same development cycle supersedes it, so the
instructions lead directly from the baseline release to the final behavior.

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
