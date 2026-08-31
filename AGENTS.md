<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai working contract

This file contains the repository-wide constraints for humans and agents. Read
the nearest package or sample README for its entry point and canonical-doc
links. User-facing docs live only under `docs/source/`.

## Repository map

```text
client-samples/  Platform clients
agent-sdk/       Agent runtime, hub IPC, model, tool, and voice libraries
agent-samples/   Runnable agent stacks
services/        Hub, model servers, and typed capability services
utils/           Launcher, logging, VAD, vLLM, and voice-gate utilities
tests/           Cross-package and integration tests
docs/source/     User and contributor documentation
skills/          Skill bank that sets coding agents up
```

## Architecture boundaries

- One hub serves many clients and agents. It fans inbound messages to subscribed
  `ProcessorEndpoint`s and routes return traffic only to the originating client.
- Agents communicate with the hub through `xr_ai_hub` IPC. LiveKit is internal
  to the hub and must not appear in agent APIs.
- The hub owns client readiness. Agents report only their own `_agent.status`.
  Readiness participation is opt-in with `announces_readiness=True`, applies
  only to subscribed participants, and must never be published directly by a
  worker.
- `agent-sdk/xr-ai-hub/` depends only on `pyzmq` and `msgpack`.
- All model HTTP goes through the typed services and factories in
  `xr_ai_models`. Do not add vendor SDKs or hand-written model HTTP clients to
  workers or services. One scoped exception: Riva speech NIMs expose only
  gRPC, so the optional `riva` extra adds `nvidia-riva-client` behind
  `kind: riva_grpc`, with the import deferred into `make_stt`/`make_tts`;
  the base install stays httpx-only.
- Workers never import `device_io_hub` or `xr_ai_launcher`. Use public SDK
  packages and task-specific libraries.
- Tools are native, in-process `Tool` or `AsyncTool` instances from
  `xr_ai_tools`. Every execution passes through NeMo Relay. The repository does
  not ship MCP compatibility servers.
- Agents call another agent's tools directly with `execute()` or `stream()`.
  `xr_ai_runtime` provides typed publish/fan-out and owns only its delivery
  tasks. Agents own their state, resources, lifecycle, background tasks, and
  concurrency policy.
- Application-specific capabilities stay with their application. Shared
  process boundaries use typed msgpack over ZMQ services, not MCP.
- Image selection and visual inference are separate tools. Selection returns
  lightweight image references; single-image, multi-image, and timestamped
  video-frame query tools resolve them through `VLMService`. Raw media remains
  on the hub path and out of tool results. Latest recorded windows need only a
  duration; historical frame and video selection share an absolute `start_us`.

The authoritative dependency graph and enforced package limits are in
[`DEPENDENCIES.md`](DEPENDENCIES.md).

## Process and sample layout

Each sample contains a stdlib-style orchestrator and a separately packaged
worker. Optional application-specific capability processes sit beside them.

```text
agent-samples/<kebab-name>/
  main.py
  pyproject.toml
  README.md
  yaml/
  worker/
    pyproject.toml
    <snake_name>_worker/
      __init__.py
      __main__.py
```

- Orchestrators depend on `xr-ai-launcher`, declare `PROCESSES`, and call
  `run_stack`.
- Stack items start in declaration order; members of a `Parallel` item start
  concurrently. The launcher waits for each item to signal ready, and fails the
  stack if any process exits.
- `device_io_hub` always runs as its own process.
- Raw IPC workers keep hub addresses as module constants, use async callbacks,
  cancel tasks before `ProcessorEndpoint.stop()` and `.close()`, and move
  CPU-bound work to an executor.
- Voice workers delegate readiness, signals, pipeline cancellation, and cleanup
  to `VoiceAgent`; its media session is private.
- New worker code is a named package with relative internal imports and an
  explicit `__main__.py`.

Refer to [Adding a sample](docs/source/guides/adding-a-sample.md) and the
[`simple-vlm-example`](agent-samples/simple-vlm-example/README.md) reference.

## Change contract

- Public Python API names, signatures, types, defaults, and field behavior live
  in `__all__`, declarations, and co-located docstrings; Sphinx generates their
  reference pages. Update narrative README or `docs/source/` content only when
  concepts, workflows, operations, or architecture change. Breaking changes
  also require a migration entry.
- Top-level sample commands and options live in their `[project.scripts]` and
  `argparse` declarations; Sphinx generates the user-facing CLI reference.
- Sample configuration values and field guidance live in checked-in YAML/JSON
  and adjacent YAML comments. Files under a top-level sample's `yaml/` tree or
  beside a direct capability subproject are generated into the config reference.
- After any `pyproject.toml` change, run
  `uv run --script .github/scripts/generate_dependency_map.py`; the pre-commit hook
  normally regenerates the Python inventory automatically and CI rejects drift.
  Do not hand-edit the generated section in `DEPENDENCIES.md`. Regenerate the
  affected project's gitignored `uv.lock` locally to verify resolution.
- The root `uv.toml` limits registry artifacts to the last dependency
  qualification timestamp. Advance it only in a dedicated dependency refresh,
  resolve every project's gitignored `uv.lock` with
  `uv --config-file uv.toml lock --upgrade --project <directory>` from the
  repository root, and run the full test suite. uv stops upward config discovery
  at a nearer `[tool.uv]` table; most nested projects define one through
  `[tool.uv.sources]`, so pass the root config explicitly. Do not commit
  generated lockfiles.
- Never put API keys or tokens in source files. Use environment variables or
  the credential store documented in `docs/source/getting_started/credentials.md`.
- Do not add an abstraction until two concrete use cases need it.
- When a sample's behavior is driven by an LLM prompt, changing a rule in its
  `system.txt` requires a corresponding eval case. Do not reuse worked-example
  specifics in the eval fixture.
- New source files need the repository SPDX header. File-type rules and
  exceptions are in [SPDX headers](docs/source/guides/spdx-headers.md).
- Preserve unrelated work in a dirty tree. Never use destructive Git commands
  to discard user changes.

## Pull request development

- Define the smallest independently useful outcome before editing. Keep the
  implementation, tests, and documentation required for that outcome together,
  but leave unrelated cleanup, redesigns, and adjacent improvements out.
- When a pull request starts an intentional series, create one issue for each
  remaining independently reviewable outcome, assign it to the authenticated
  GitHub user, and link it from the pull request. Do not use future work to
  excuse an incomplete current change.
- Write the pull request description around four facts: the problem being
  solved, the solution in this diff, deliberate scope boundaries and linked
  follow-ups, and validation performed. Keep it current as the diff changes so
  reviewers can distinguish an intentional tradeoff from an omission.
- Before requesting review, perform at least one isolated self-review of the
  complete merge-base diff. Re-read the stated outcome without relying on
  implementation notes, inspect every changed line for unnecessary churn and
  failure modes, and confirm that tests and documentation match the behavior.
- Treat review feedback as evidence to understand, not an implementation order.
  Reproduce the concern, apply the smallest correct high-value fix that fits the
  pull request, and state the disposition. Decline unrelated work or a requested
  redesign, or move it to a linked follow-up when it is accepted as planned
  work, instead of silently expanding scope.

The operational authoring workflow is in
[`gh-develop-xr-ai`](skills/gh-develop-xr-ai/SKILL.md).

## Comments and documentation

Comments explain a non-obvious invariant or failure mode. Do not narrate the
code, debugging history, rejected alternatives, or plans. Keep architectural
docs about the current system; issue trackers and Git history hold proposals
and past decisions.

Follow the authoritative [documentation style](docs/source/guides/documentation-style.md)
for customer-facing Markdown and reStructuredText. Match the surrounding
document and keep edits scoped.

`docs/source/` is the canonical home for user-facing concepts, architecture,
workflows, operations, troubleshooting, and reference material. Do not duplicate
that content in a README. A README is a concise repository entry point: state
what the directory contains, link to the canonical documentation, and keep only
the minimum local information needed before following that link. A top-level
sample README under `agent-samples/` additionally keeps the exact commands needed
to start that sample. State the sample directory as the working directory and
write every command relative to that directory, including commands that invoke a
sibling sample such as `../model-servers`. It also includes a compact
configuration entry point: name the sample-owned YAML or JSON files, describe
the common settings they own, and show one representative edit. Put exhaustive
field descriptions, precedence, path resolution, architecture, behavior,
artifact contracts, extension instructions, and eval methodology in the
sample's `docs/source/` guide. Nested READMEs may contain only artifact-local
instructions that are not useful in the published documentation.

When documentation exists in both locations, consolidate it into
`docs/source/` and replace the README copy with a link. When moving README
content, preserve any still-valid operational detail in the destination page;
do not delete unique guidance merely to shorten the README. Package API details
remain in declarations and co-located docstrings and are rendered into the
generated Python reference rather than repeated in either narrative location.

Use these canonical references when working in their area:

| Topic | Document |
|---|---|
| Architecture | [`docs/source/overview/architecture.md`](docs/source/overview/architecture.md) |
| Processes | [`docs/source/components/launcher-and-process-model.md`](docs/source/components/launcher-and-process-model.md) |
| Agent SDK | [`docs/source/components/agent-sdk.md`](docs/source/components/agent-sdk.md) |
| AI services | [`docs/source/components/ai-services.md`](docs/source/components/ai-services.md) |
| Credentials | [`docs/source/getting_started/credentials.md`](docs/source/getting_started/credentials.md) |
| Networking | [`docs/source/getting_started/networking.md`](docs/source/getting_started/networking.md) |
| Testing | [`docs/source/guides/testing.md`](docs/source/guides/testing.md) |
| Documentation style | [`docs/source/guides/documentation-style.md`](docs/source/guides/documentation-style.md) |
| Troubleshooting | [`docs/source/guides/troubleshooting.md`](docs/source/guides/troubleshooting.md) |
| xr-render-demo | [`docs/source/reference/xr-render-demo.md`](docs/source/reference/xr-render-demo.md) |
