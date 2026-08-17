<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai working contract

This file contains the repository-wide constraints for humans and agents. Read
the nearest package or sample README for local details. User-facing docs live
only under `docs/source/`.

## Repository map

```text
client-samples/  Platform clients
agent-sdk/       Agent runtime, hub IPC, model, tool, and voice libraries
agent-samples/   Runnable agent stacks
services/        Hub, model servers, and typed capability services
utils/           Launcher, logging, VAD, vLLM, and voice-gate utilities
tests/           Cross-package and integration tests
docs/source/     User and contributor documentation
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
  workers or services.
- Workers never import `xr_media_hub` or `xr_ai_launcher`. Use public SDK
  packages and task-specific libraries.
- Tools are native, in-process `Tool` or `AsyncTool` instances from
  `xr_ai_tools`. Every execution passes through NeMo Relay. The repository does
  not ship MCP compatibility servers.
- Agents call another agent's tools directly with `execute()` or `stream()`.
  `xr_ai_runtime` provides typed publish/fan-out and owns only its delivery
  tasks. Agents own their state, resources, lifecycle, background tasks, and
  concurrency policy.
- Application-specific capabilities stay with their application. Shared
  process boundaries use typed msgpack/ZMQ services, not MCP.
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
- Processes start serially, touch their `--ready-file` when ready, and fail the
  stack if any process exits.
- `xr_media_hub` always runs as its own process.
- Raw IPC workers keep hub addresses as module constants, use async callbacks,
  cancel tasks before `ProcessorEndpoint.stop()` and `.close()`, and move
  CPU-bound work to an executor.
- Voice workers delegate readiness, signals, pipeline cancellation, and cleanup
  to `VoiceAgent`; its media session is private.
- New worker code is a named package with relative internal imports and an
  explicit `__main__.py`.

See [Adding a sample](docs/source/guides/adding-a-sample.md) and the
[`simple-vlm-example`](agent-samples/simple-vlm-example/README.md) reference.

## Change contract

- Public Python API names, signatures, types, defaults, and field behavior live
  in `__all__`, declarations, and co-located docstrings; Sphinx generates their
  reference pages. Update narrative README or `docs/source/` content only when
  concepts, workflows, operations, or architecture change. Breaking changes
  also require a migration entry.
- Any `pyproject.toml` change must update `DEPENDENCIES.md`; regenerate the
  affected project's gitignored `uv.lock` locally to verify resolution.
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

## Comments and documentation

Comments explain a non-obvious invariant or failure mode. Do not narrate the
code, debugging history, rejected alternatives, or plans. Keep architectural
docs about the current system; issue trackers and Git history hold proposals
and past decisions.

Use these canonical references when working in their area:

| Topic | Document |
|---|---|
| Architecture | [`docs/source/overview/architecture.md`](docs/source/overview/architecture.md) |
| Processes | [`docs/source/components/launcher-and-process-model.md`](docs/source/components/launcher-and-process-model.md) |
| Agent SDK | [`docs/source/components/agent-sdk.md`](docs/source/components/agent-sdk.md) |
| AI services | [`docs/source/components/ai-services.md`](docs/source/components/ai-services.md) |
| Credentials | [`docs/source/getting_started/credentials.md`](docs/source/getting_started/credentials.md) |
| Networking | [`docs/source/getting_started/networking.md`](docs/source/getting_started/networking.md) |
| Troubleshooting | [`docs/source/guides/troubleshooting.md`](docs/source/guides/troubleshooting.md) |
| xr-render-demo | [`docs/source/reference/xr-render-demo.md`](docs/source/reference/xr-render-demo.md) |
